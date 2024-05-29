import os
import sys
import tempfile
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

from dask.distributed import Client, as_completed
from datasets import Dataset
from rich import print as rprint
from rich import traceback
from rich.progress import Progress
from sqlitedict import SqliteDict
from transformers import AutoTokenizer

import utils
from data.audio_processing import AudioUtils, CodecProcessor, SpectrogramProcessor
from data.beatmap_parser import parse_beatmap
from data.beatmap_tokenizer import build_header, build_segments
from data.utils import ProcessingError

from config import BeatSaberConfig

traceback.install()


class CodecActor:
    """Singleton worker for encoding audio on the GPU."""

    def __init__(self, config):
        self.config = config
        self.codec_processor = CodecProcessor()

    def encode(self, mp3_path, bpm, batch_size):
        return self.codec_processor.encode(mp3_path, bpm, batch_size)


class TokenizerActor:
    """Singleton worker for tokenizing and building the vocabulary."""

    token_counts = Counter()
    tokenizer = None

    def __init__(self, config):
        TokenizerActor.tokenizer = AutoTokenizer.from_pretrained(config.lm.model)

    def tokenize(self, text_tokens):
        for segment in text_tokens:
            for token in segment:
                if token not in self.token_counts:
                    self.tokenizer.add_tokens([token])

                self.token_counts[token] += 1

        encoded = self.tokenizer(
            text_tokens,
            add_special_tokens=False,
            is_split_into_words=True,
            padding=False,
        )

        return encoded["input_ids"]

    def save(self, path):
        self.tokenizer.save_pretrained(path)


def load_metadata(data_path, ids=[]):
    """Load song data from metadata.sqlite, created by scrape.py."""

    db_path = data_path / "metadata.sqlite"
    with SqliteDict(db_path, autocommit=True) as db:
        metadata = db.get("maps", {})

    if ids:
        metadata = {k: v for k, v in metadata.items() if v["id"] in ids}

    # Superceded beatmaps often have [Old Version] in the title.
    metadata = {k: v for k, v in metadata.items() if "old version" not in v["name"].lower()}

    metadata = sorted(metadata.items(), key=lambda x: x[1]["stats"]["score"], reverse=True)
    return OrderedDict(metadata)


def process_info(path, root):
    """Process the info.dat file within a song zip."""

    data = utils.load_json(path)

    if data.get("_songTimeOffset") != 0:
        raise ProcessingError("Non-zero time offset")

    if not (egg_path := data.get("_songFilename")):
        raise ProcessingError("_songFilename not present")

    egg_path = os.path.join(root, egg_path)
    if not os.path.exists(egg_path):
        raise ProcessingError("Song file does not exist")

    difficulty_set = next(
        iter(
            set_.get("_difficultyBeatmaps", [])
            for set_ in data.get("_difficultyBeatmapSets", [])
            if set_.get("_beatmapCharacteristicName") == "Standard"
        ),
        None,
    )
    if difficulty_set is None:
        raise ProcessingError("Song file does not exist")

    difficulties = [d for d in difficulty_set if d.get("_difficulty") in utils.DIFFICULTIES.keys()]
    if not difficulties:
        raise ProcessingError("No usable difficulties")

    # Skip anything too slow or too fast.
    bpm = data.get("_beatsPerMinute")
    if not 70 <= bpm <= 300:
        raise ProcessingError("BPM not in range")

    # Resolve beatmap paths.
    beatmaps = {}
    for difficulty in difficulties:
        beatmaps[difficulty.get("_difficulty")] = os.path.join(
            root, difficulty.get("_beatmapFilename")
        )

    return beatmaps, egg_path, bpm


def create_sample(
    config,
    beatmap_path,
    difficulty,
    bpm,
    rating,
    codec_embeddings,
    spectrograms,
    tokenizer,
):
    """Create a single training sample for a given difficulty."""

    beatmap, error = parse_beatmap(
        beatmap_path,
        num_segments=len(spectrograms),
        bpm=bpm,
        difficulty=difficulty,
        rating=rating,
    )
    if error:
        raise ProcessingError(error)

    # Tokenize beatmap.
    header = build_header(beatmap)
    segments = build_segments(
        beatmap,
        100,
        config.data.note_format,
    )

    if header is None or segments is None:
        raise ProcessingError("Could not tokenize beatmap.")

    header_ids = tokenizer.tokenize([header]).result()[0]
    segment_ids = tokenizer.tokenize(segments).result()

    return {
        "codec": codec_embeddings,
        "spectrograms": spectrograms,
        "header": header_ids,
        "segments": segment_ids,
        "header_tokens": header,
        "segment_tokens": segments,
    }


def process_zip(zip_path, metadata, config, codec, tokenizer):
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)
        except zipfile.BadZipFile:
            return [], [ProcessingError("Bad zip file")]

        # Load metadata from info.dat.
        info_path = None
        for entry in os.scandir(tmp_dir):
            if entry.is_file() and entry.name.lower() == "info.dat":
                info_path = entry.path
                break

        try:
            beatmaps, egg_path, bpm = process_info(info_path, tmp_dir)
        except ProcessingError as e:
            return [], [e]

        # Calculate embeddings.
        mp3_path = os.path.join(tmp_dir, "song.mp3")
        try:
            AudioUtils.convert(egg_path, mp3_path)
            codec_embeddings = codec.encode(
                mp3_path,
                bpm,
                config.data.codec_batch_size,
            ).result()
            spectrograms = SpectrogramProcessor.encode(mp3_path, bpm, config.conformer.mel_bands)
        except Exception as e:
            return [], [e]

        if len(spectrograms) != len(codec_embeddings):
            raise ProcessingError(
                f"Codec and spectrogram lengths do not match: {len(codec_embeddings)}, {len(spectrograms)}"
            )

        beatmaps = sorted(
            beatmaps.items(),
            key=lambda x: utils.DIFFICULTIES[x[0]],
            reverse=True,
        )
        samples, errors = [], []
        for difficulty, beatmap_path in beatmaps:
            try:
                sample = create_sample(
                    config,
                    beatmap_path,
                    difficulty,
                    bpm,
                    metadata["stats"]["score"],
                    codec_embeddings,
                    spectrograms,
                    tokenizer,
                )
            except ProcessingError as e:
                sample = None
                errors.append(e)

            if sample:
                samples.append(sample)

                if config.data.only_highest_difficulty:
                    break

        return samples, errors


def process_zips(zip_files, metadata, config, client, codec, tokenizer, shards_dir):
    shard_paths = []
    num_samples = 0

    def save_shard():
        nonlocal num_samples
        dataset = Dataset.from_list(samples)
        num_samples += len(samples)
        samples.clear()
        shard_path = shards_dir / str(len(shard_paths))
        shard_paths.append(shard_path)
        dataset.save_to_disk(str(shard_path))

    with Progress() as progress:
        progress_task = progress.add_task(
            "[bold blue]Processing zips into training samples...",
            total=len(zip_files),
        )

        samples = []
        errors = Counter()
        num_workers = config.data.preprocess_workers

        # Submit an initial batch of tasks.
        futures = [
            client.submit(
                process_zip,
                zip_file,
                metadata[zip_file.stem],
                config,
                codec,
                tokenizer,
            )
            for zip_file in zip_files[:num_workers]
        ]
        remaining_files = zip_files[num_workers:]

        while futures:
            for future in as_completed(futures):
                try:
                    new_samples, new_errors = future.result()
                    samples.extend(new_samples)
                    errors.update([str(e) for e in new_errors])

                    if len(samples) >= config.data.shard_size:
                        save_shard()

                    progress.advance(progress_task)
                except Exception as e:
                    errors.update([str(e)])

                futures.remove(future)

                # Submit a new task for the next file in the list if available
                if remaining_files:
                    next_zip = remaining_files.pop(0)
                    new_future = client.submit(
                        process_zip,
                        next_zip,
                        metadata[next_zip.stem],
                        config,
                        codec,
                        tokenizer,
                    )
                    futures.append(new_future)

    if samples:
        save_shard()

    return shard_paths, num_samples, errors


def main():
    client = Client()
    config = BeatSaberConfig()
    codec = client.submit(CodecActor, config, actor=True).result()
    tokenizer = client.submit(TokenizerActor, config, actor=True).result()

    if len(sys.argv) != 2:
        rprint("[red]Usage: python -m data.preprocess <data_path>")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    zip_files = list((data_path / "zips").glob("*.zip"))
    rprint(f"[green]Found {len(zip_files)} zips.")

    metadata = load_metadata(data_path)
    zip_files = [z for z in zip_files if z.stem in metadata]
    if config.data.max_songs != -1:
        zip_files = zip_files[: config.data.max_songs]
    rprint(f"[green]Now {len(zip_files)} zips after filtering for metadata.")

    shard_paths, num_samples, errors = process_zips(
        zip_files,
        metadata,
        config,
        client,
        codec,
        tokenizer,
        data_path / "shards",
    )
    rprint(f"[green]Processed {num_samples} samples.")
    rprint(f"[yellow]Errors: {errors}")

    tokenizer.save(data_path / "tokenizer").result()
    rprint(f"[green]Saved tokenizer.")

    global_mean, global_std = SpectrogramProcessor.normalize(shard_paths)
    rprint(f"[bold green]Saved dataset: {global_mean}, {global_std}")


if __name__ == "__main__":
    main()
