import sys

import librosa
import torch
from rich import traceback
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

from config import BeatSaberConfig
from data.audio_processing import CodecProcessor
from data.beatmap_tokenizer import quantize_bpm
from model.model import BeatSaberModel
from train import load_dataset

traceback.install()
config = BeatSaberConfig()

tokenizer = AutoTokenizer.from_pretrained(config.lm.model)


def generate(mp3_path):
    # y, sr = librosa.load(mp3_path)
    # bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    # bpm = round(bpm[0])
    # quantized_bpm = quantize_bpm(bpm)
    # header = f"<header> Difficulty: expert-plus | BPM level: {quantized_bpm} | Rating: 9 | Note density level: 6 </header>"
    # processor = CodecProcessor()
    # audio_embeds = processor.encode(mp3_path, bpm, 8)
    # audio_embeds = torch.tensor(audio_embeds).to("cuda", dtype=torch.bfloat16)

    model = BeatSaberModel.from_pretrained("/model/checkpoint-450").to(torch.bfloat16)
    model.eval()

    sample = load_dataset(config, "/data")["train"][0]
    sample["header"] = sample["retokenized_header"]
    sample["segments"] = sample["retokenized_segments"]
    print(tokenizer.decode(sample["header"]))

    audio_embeds = sample["codec_embeddings"]
    _ = model([sample])
    header_tokens = tokenizer.decode(sample["header"], skip_special_tokens=True)
    tokens = model.generate(audio_embeds, header_tokens)
    print(tokens)
    print(tokenizer.decode(tokens))

    # create_zip(
    #     "/data/new_beatmap.zip",
    #     mp3_path,
    #     {"ExpertPlus": tokens[0]},
    #     bpm,
    #     "Title",
    #     "Artist",
    # )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m infer <mp3_path>")
        sys.exit(1)

    mp3_path = sys.argv[1]
    with torch.device("cuda"):
        generate(mp3_path)
