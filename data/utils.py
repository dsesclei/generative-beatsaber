import json

import chardet

DIFFICULTIES = {"ExpertPlus": 4, "Expert": 3, "Hard": 2, "Normal": 1, "Easy": 0}


class ProcessingError(Exception):
    pass


def load_json(path):
    with open(path, "rb") as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)["encoding"]

    with open(path, encoding=encoding) as f:
        return json.load(f)
