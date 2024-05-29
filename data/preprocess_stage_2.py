import os
import sys
from pathlib import Path

from dask.distributed import Client, as_completed
from datasets import Dataset, load_from_disk
from rich import print as rprint
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from transformers import AutoTokenizer

from config import BeatSaberConfig
from data.retokenize import convert_header, convert_segment

config = BeatSaberConfig()
tokenizer = AutoTokenizer.from_pretrained(config.lm.model, token=os.environ.get("HF_TOKEN"))


def tokenize(text):
    return tokenizer(text, padding=False, add_special_tokens=False)["input_ids"]


def retokenize(batch):
    batch["retokenized_header"] = [
        tokenize(convert_header(header)) for header in batch["header_tokens"]
    ]
    batch["retokenized_segments"] = [
        [tokenize(convert_segment(segment)) for segment in segments]
        for segments in batch["segment_tokens"]
    ]
    num_audio_tokens = [2 * config.model.num_audio_tokens * len(s) for s in batch["spectrograms"]]
    batch["length"] = [
        2 + num_t + len(header) + 2 * len(segments) + sum(len(seg) for seg in segments)
        for num_t, header, segments in zip(
            num_audio_tokens, batch["retokenized_header"], batch["retokenized_segments"]
        )
    ]
    return batch


def fits_in_context(num_audio_tokens, header, segments):
    length = (
        2 + num_audio_tokens + len(header) + 2 * len(segments) + sum(len(seg) for seg in segments)
    )
    return length <= config.lm.context_size


def filter_batch(batch):
    num_audio_tokens = [2 * config.model.num_audio_tokens * len(s) for s in batch["spectrograms"]]
    return [
        fits_in_context(num_t, header, segments)
        for num_t, header, segments in zip(
            num_audio_tokens, batch["retokenized_header"], batch["retokenized_segments"]
        )
    ]


def process_shard(shard_path):
    shard = load_from_disk(str(shard_path))
    original_length = len(shard)
    shard = shard.map(retokenize, batched=True, batch_size=20)

    valid_indices = [i for i, valid in enumerate(filter_batch(shard)) if valid]
    dropped_indices = [i for i in range(original_length) if i not in valid_indices]

    dropped_data = {
        "text": [
            f"{convert_header(shard['header_tokens'][i])} {' seg '.join(convert_segment(segment) for segment in shard['segment_tokens'][i])}"
            for i in dropped_indices
        ]
    }

    if dropped_data["text"]:
        dropped_dataset = Dataset.from_dict(dropped_data)
        dropped_dataset.save_to_disk(str(shard_path.parent.parent / "dropped" / shard_path.name))

    valid_shard = shard.select(valid_indices)
    valid_shard.save_to_disk(str(shard_path.parent.parent / "postprocessed" / shard_path.name))

    return len(dropped_indices)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        rprint("[red]Usage: python -m data.preprocess_stage_2 <data_path>")
        sys.exit(1)
    data_path = Path(sys.argv[1])
    shard_paths = list((data_path / "dataset").iterdir())

    client = Client()

    num_workers = 32

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing shards...", total=len(shard_paths))

        futures = [client.submit(process_shard, shard) for shard in shard_paths[:num_workers]]
        remaining_shards = shard_paths[num_workers:]
        num_dropped = 0

        while futures:
            for future in as_completed(futures):
                num_dropped += future.result()
                progress.update(task, advance=1)
                futures.remove(future)
                if remaining_shards:
                    next_shard = remaining_shards.pop(0)
                    new_future = client.submit(process_shard, next_shard)
                    futures.append(new_future)

    rprint(f"[green]Dropped {num_dropped} records.")
    client.close()
