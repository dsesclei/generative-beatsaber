#!/usr/bin/env ipython
import re


directions = {
    "0": "up",
    "1": "down",
    "2": "left",
    "3": "right",
    "4": "up-left",
    "5": "up-right",
    "6": "down-left",
    "7": "down-right",
    "8": "any",
}

row_locations = {
    "0": "bottom",
    "1": "middle",
    "2": "top",
}

col_locations = {
    "0": "far-left",
    "1": "left",
    "2": "right",
    "3": "far-right",
}

patterns = {
    "note": re.compile(r"^NOTE_(\d)_(\d)_(\d)_(\d)"),
    "time": re.compile(r"^TIME_(\d+)"),
    "bpm": re.compile(r"^BPM_(\d)"),
    "npb": re.compile(r"^NPB_(\d)"),
    "rating": re.compile(r"^RATING_(\d)"),
    "difficulty": re.compile(r"^DIFFICULTY_([A-Z]+)"),
}


def convert_header(tokens):
    all_text = []
    for token in tokens:
        # Strip [brackets]
        token = token[1:-1]
        if m := patterns["bpm"].match(token):
            text = f"BPM level: {m[1]}"
        elif m := patterns["npb"].match(token):
            text = f"Note density level: {m[1]}"
        elif m := patterns["rating"].match(token):
            text = f"Rating: {m[1]}"
        elif m := patterns["difficulty"].match(token):
            difficulty = "expert-plus" if m[1] == "EXPERTPLUS" else m[1].lower()
            text = f"Difficulty: {difficulty}"
        else:
            match token:
                case "HEADER":
                    text = "<header>"
                case "END_HEADER":
                    text = "</header>"
                case "BOMBS":
                    text = "bombs"
                case "WALLS":
                    text = "walls"
                case _:
                    print("Unidentified", token)
                    text = ""

        if "header" not in text and "Difficulty" not in text:
            text = "| " + text
        all_text.append(text)
    return " ".join(all_text)


def convert_segment(tokens):
    all_text = []
    for token in tokens:
        # Strip [brackets]
        token = token[1:-1]
        if m := patterns["note"].match(token):
            color = "red" if m[1] == "0" else "blue"
            row, col = row_locations[m[2]], col_locations[m[3]]
            direction = directions[m[4]]
            text = f"{color} {row} {col} {direction}"
        elif m := patterns["time"].match(token):
            text = f"{m[1]}%"
        else:
            match token:
                # Ignore start and end, add in model.
                case "SEG":
                    text = ""
                case "END_SEG":
                    text = ""
                case "SHORT_GAP":
                    text = "gap"
                case "BOMB":
                    text = "bombs"
                case "WALL":
                    text = "walls"
                case _:
                    print("Unidentified", token)
                    text = ""

        all_text.append(text.strip())
    return " ".join(all_text)
