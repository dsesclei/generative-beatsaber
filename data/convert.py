import json
import re
import zipfile

from audio_processing import AudioUtils


def convert_segment(segment, beats_per_segment, idx):
    note_pattern = r"NOTE_(?P<color>\d)_(?P<row>\d)_(?P<column>\d)_(?P<direction>\d)"
    notes = []
    beat_delta = 0

    for token in segment:
        token = token.strip("[]")

        if token.startswith("TIME"):
            time_pos = int(token.split("_")[1])
            beat_delta = beats_per_segment * (time_pos / 100)
        elif token.startswith("NOTE"):
            if match := re.match(note_pattern, token):
                note = {
                    "c": int(match.group("color")),
                    "y": int(match.group("row")),
                    "x": int(match.group("column")),
                    "d": int(match.group("direction")),
                    "b": round(idx * beats_per_segment + beat_delta, 3),
                }
            else:
                raise ValueError(f"Invalid note token: {token}")
            notes.append(note)
    return notes


def create_zip(zip_path, audio_path, difficulties, bpm, title, artist):
    beats_per_segment = AudioUtils.beats_per_segment(bpm)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        # Add audio file.
        zipf.write(audio_path, arcname="song.egg")

        # Add Info.dat.
        info_json = create_info_json(title, artist, bpm, difficulties)
        zipf.writestr("Info.dat", json.dumps(info_json))

        # Add beatmap .dat files.
        for difficulty_name, segments in difficulties.items():
            difficulty_json = create_difficulty_json(segments, beats_per_segment)
            zipf.writestr(f"{difficulty_name}Standard.dat", json.dumps(difficulty_json))


def create_difficulty_json(segments, beats_per_segment):
    data = {
        "version": "3.2.0",
        "bpmEvents": [],
        "rotationEvents": [],
        "bombNotes": [],
        "obstacles": [],
        "sliders": [],
        "burstSliders": [],
        "waypoints": [],
        "basicBeatmapEvents": [],
        "colorBoostBeatmapEvents": [],
        "lightColorEventBoxGroups": [],
        "lightRotationEventBoxGroups": [],
        "lightTranslationEventBoxGroups": [],
        "basicEventTypesWithKeywords": {},
        "useNormalEventsAsCompatibleEvents": False,
    }

    notes = [convert_segment(segment, beats_per_segment, i) for i, segment in enumerate(segments)]
    notes = [note for segment in notes for note in segment]

    data["colorNotes"] = notes
    return data


def create_info_json(title, artist, bpm, difficulty_names):
    difficulty_ranks = {
        "Easy": 1,
        "Normal": 3,
        "Hard": 5,
        "Expert": 7,
        "ExpertPlus": 9,
    }

    info = {
        "_version": "2.1.0",
        "_songName": title,
        "_songSubName": "",
        "_songAuthorName": artist,
        "_levelAuthorName": "Automapper",
        "_beatsPerMinute": bpm,
        "_shuffle": 0,
        "_shufflePeriod": 0,
        "_previewStartTime": 0,
        "_previewDuration": 14,
        "_songFilename": "song.egg",
        "_coverImageFilename": "cover.png",
        "_environmentName": "TheSecondEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {
                        "_difficulty": difficulty,
                        "_difficultyRank": difficulty_ranks[difficulty],
                        "_beatmapFilename": f"{difficulty}Standard.dat",
                        "_noteJumpMovementSpeed": 0,
                        "_noteJumpStartBeatOffset": 0,
                    }
                    for difficulty in difficulty_names
                ],
            }
        ],
    }

    return info
