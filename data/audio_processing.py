import math
import os
import subprocess

import dac
import librosa
import numpy as np
import torch
from datasets import load_from_disk

from config import BeatSaberConfig

config = BeatSaberConfig()


class CodecProcessor:
    """Generates embeddings using descript-audio-codec."""

    DAC_SCALING_FACTOR = 1 / 512  # Downsample factor for DAC.

    def __init__(self):
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path).to("cuda")

    def encode(self, audio_path, bpm, batch_size):
        segments = AudioUtils.load_and_segment_audio(audio_path, bpm)
        target_size = math.ceil(AudioUtils.samples_per_segment(bpm) * self.DAC_SCALING_FACTOR)
        audio_embeddings = []

        for start in range(0, len(segments), batch_size):
            batch_segments = segments[start : start + batch_size]
            embeddings = self.encode_batch(batch_segments, target_size)
            audio_embeddings.extend(embeddings[: len(batch_segments)])

        return audio_embeddings

    def encode_batch(self, segments, target_size):
        tensor = torch.tensor(segments).unsqueeze(1).to(self.model.device)
        preprocessed = self.model.preprocess(tensor, AudioUtils.SAMPLE_RATE)
        _, c, _, _, _ = self.model.encode(preprocessed)
        c = torch.nn.functional.pad(c, (0, 0, 0, max(0, target_size - c.shape[-1])), "constant", 0)
        return c.cpu().numpy()


class SpectrogramProcessor:
    """Generates log mel-spectrogram embeddings on CPU."""

    @staticmethod
    def encode(audio_path, bpm, n_mels):
        segments = AudioUtils.load_and_segment_audio(audio_path, bpm)
        mels = [
            librosa.feature.melspectrogram(
                y=segment,
                sr=AudioUtils.SAMPLE_RATE,
                n_fft=1024,
                hop_length=441,
                n_mels=n_mels,
            )
            for segment in segments
        ]

        # Transpose to [time, mels] so the variable time dim is first.
        log_mels = [librosa.power_to_db(mel, ref=np.max).T for mel in mels]
        return log_mels

    @staticmethod
    def gather_global_stats(shard_paths):
        total_sum, total_sum_sq, total_count = 0.0, 0.0, 0
        for path in shard_paths:
            shard = load_from_disk(str(path))
            for sample in shard:
                flat_spectra = np.ravel(sample["spectrograms"])
                total_sum += np.sum(flat_spectra)
                total_sum_sq += np.sum(np.square(flat_spectra))
                total_count += flat_spectra.size
        mean = total_sum / total_count
        std = np.sqrt(total_sum_sq / total_count - mean**2)
        return mean, std

    @staticmethod
    def normalize(shard_paths):
        global_mean, global_std = SpectrogramProcessor.gather_global_stats(shard_paths)

        def normalize_sample(sample):
            normalized_spectra = [
                (spectra - global_mean) / global_std for spectra in sample["spectrograms"]
            ]
            return {"spectrograms": normalized_spectra}

        for path in shard_paths:
            shard = load_from_disk(str(path))
            normalized_shard = shard.map(
                normalize_sample,
                batched=True,
                batch_size=10,
                num_proc=config.data.preprocess_workers,
            )
            normalized_dir = os.path.join(path.parent.parent, "dataset", path.name)
            os.makedirs(normalized_dir, exist_ok=True)
            normalized_shard.save_to_disk(str(normalized_dir))

        return global_mean, global_std


class AudioUtils:
    SAMPLE_RATE = 44100

    @staticmethod
    def samples_per_beat(bpm):
        beat_duration = 60.0 / bpm
        return int(beat_duration * AudioUtils.SAMPLE_RATE)

    @staticmethod
    def samples_per_segment(bpm):
        return AudioUtils.samples_per_beat(bpm) * AudioUtils.beats_per_segment(bpm)

    @staticmethod
    def beats_per_segment(bpm):
        beat_duration = 60.0 / bpm
        return round(beat_duration * config.data.segment_length)

    @staticmethod
    def load_and_segment_audio(audio_path, bpm):
        y, _ = librosa.load(audio_path, sr=AudioUtils.SAMPLE_RATE)
        samples_per_segment = AudioUtils.samples_per_segment(bpm)
        padding_length = (samples_per_segment - len(y) % samples_per_segment) % samples_per_segment
        y_padded = np.pad(y, (0, padding_length), mode="constant")
        return y_padded.reshape(-1, samples_per_segment)

    @staticmethod
    def convert(source_path, output_path):
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                source_path,
                "-ar",
                str(AudioUtils.SAMPLE_RATE),
                "-ac",
                "1",
                "-filter:a",
                "loudnorm",
                "-sample_fmt",
                "s16",
                output_path,
                "-loglevel",
                "quiet",
                "-y",
            ],
            check=True,
        )
