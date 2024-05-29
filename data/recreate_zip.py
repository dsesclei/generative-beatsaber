import os
import sys
import tempfile
import zipfile

from dask.distributed import Client
from rich import traceback

from convert import create_zip
from preprocess import CodecActor, TokenizerActor, process_info, process_zip
from ..config import BeatSaberConfig

traceback.install()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recreate_zip.py <zip_path>")
        sys.exit(1)
    zip_path = sys.argv[1]

    config = BeatSaberConfig()
    client = Client()
    codec = client.submit(CodecActor, config, actor=True).result()
    tokenizer = client.submit(TokenizerActor, config, actor=True).result()
    samples, errors = process_zip(
        zip_path,
        {"stats": {"score": 0.95}},
        codec,
        tokenizer,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)

        # Load metadata from info.dat.
        info_path = None
        for entry in os.scandir(tmp_dir):
            if entry.is_file() and entry.name.lower() == "info.dat":
                info_path = entry.path
                break

        beatmaps, egg_path, bpm = process_info(info_path, tmp_dir)

        create_zip(
            "/data/new_beatmap.zip",
            egg_path,
            {"ExpertPlus": samples[0]["segment_tokens"]},
            bpm,
            "Title",
            "Artist",
        )
