from more_itertools import partition

from audio_processing import AudioUtils
from constants import BEATMAP_KEYS
from utils import ProcessingError, load_json


class BeatmapData:
    SLIDER_HEAD = "HEAD"
    SLIDER_MID = "MID"
    SLIDER_TAIL = "TAIL"

    def __init__(
        self,
        num_segments,
        bpm,
        difficulty,
        rating,
        notes,
        sliders,
        bomb_segments,
        wall_segments,
    ):
        self.beats_per_segment = AudioUtils.beats_per_segment(bpm)
        self.num_segments = num_segments
        self.bpm = bpm
        self.difficulty = difficulty
        self.rating = rating
        self.notes = notes
        self.sliders = sliders
        self.bomb_segments = bomb_segments
        self.wall_segments = wall_segments

    def slider_for_note(self, note):
        slider_key = (note["beat"], note["color"], note["row"], note["column"])

        is_head = (slider_key + (BeatmapData.SLIDER_HEAD,)) in self.sliders
        is_tail = (slider_key + (BeatmapData.SLIDER_TAIL,)) in self.sliders

        if is_head and is_tail:
            return BeatmapData.SLIDER_MID
        if is_head:
            return BeatmapData.SLIDER_HEAD
        if is_tail:
            return BeatmapData.SLIDER_TAIL

        return None


def parse_beatmap(path, num_segments, bpm, difficulty, rating):
    data = load_json(path)

    if has_bpm_events(data):
        raise ProcessingError("BPM events")

    version = data.get("_version", "") or data.get("version", "")
    if not (version.startswith("2") or version.startswith("3")):
        raise ProcessingError("Unknown version")

    if version.startswith("2"):
        # Separate out v2 bombs into a new _bombs key.
        is_bomb = lambda n: n.get("_type", -1) == 3
        data["_notes"], data["_bombs"] = partition(is_bomb, data["_notes"])

    data = normalize_keys(data, BEATMAP_KEYS[version[0]])
    if "burst_sliders" not in data:
        pass
        # TODO burst sliders.

    # Remove fake notes.
    notes = [note for note in data["notes"] if not is_fake(note)]

    # Validate.
    error = validate_notes(notes)
    if error:
        return None, error

    # Sort for a stable presentation to the model.
    notes = sorted(notes, key=lambda n: (n["beat"], n["color"], n["column"], n["row"]))

    # Destructure sliders into flat list of slider start and end points.
    slider_pairs = [create_sliders(slider) for slider in data["sliders"]]
    sliders = [slider for pair in slider_pairs for slider in pair]

    # Convert bomb/wall lists into sets of segment indices.
    beats_per_segment = AudioUtils.beats_per_segment(bpm)
    bomb_segments, wall_segments = get_hazard_segments(
        data["bombs"], data["walls"], beats_per_segment
    )

    beatmap_data = BeatmapData(
        num_segments,
        bpm,
        difficulty,
        rating,
        notes,
        sliders,
        bomb_segments,
        wall_segments,
    )

    return (beatmap_data, None)


def normalize_keys(data, keys):
    beatmap = {new: data.get(old, []) for new, old in keys["top"].items()}
    for object_type, map_objects in beatmap.items():
        object_keys = keys[object_type]
        objects = [{new: obj.get(old) for new, old in object_keys.items()} for obj in map_objects]
        beatmap[object_type] = objects

    return beatmap


def get_hazard_segments(bombs, walls, beats_per_segment):
    bomb_segments = {int(bomb["beat"]) // beats_per_segment for bomb in bombs}
    wall_segments = set()
    for wall in walls:
        beat, duration = int(wall["beat"]), int(max(0, wall.get("duration", 0)))
        wall_segments.update(
            range(beat // beats_per_segment, (beat + duration) // beats_per_segment + 1)
        )
    return bomb_segments, wall_segments


# Convert a slider into two indexable tuples, so we can easily look up if a slider is on a note later.
def create_sliders(slider):
    head = (slider["head_beat"], slider["head_row"], slider["head_column"])
    tail = (slider["tail_beat"], slider["tail_row"], slider["tail_column"])

    return [
        (slider["color"], *head, BeatmapData.SLIDER_HEAD),
        (slider["color"], *tail, BeatmapData.SLIDER_TAIL),
    ]


# Songs with variable BPMs are skipped over for now.
def has_bpm_events(data):
    if bpm_events := data.get("_BPMChanges", data.get("bpmEvents", [])):
        # Some songs have a noop BPM event at the start, which we can ignore.
        first_event_beat = bpm_events[0].get("_time", bpm_events[0].get("b", 0))
        return len(bpm_events) > 1 or first_event_beat != 0
    else:
        return False


# Notes might be fake according to this parser:
# https://github.com/AllPoland/ArcViewer/blob/main/Assets/__Scripts/BeatmapData/BeatmapDifficultyV2.cs#L70
def is_fake(note):
    is_fake_v2 = note.get("_customData", {}).get("_fake", False)
    is_fake_v3 = note.get("customData", {}).get("fake", False)

    return is_fake_v2 or is_fake_v3


def validate_notes(notes):
    # Skip suspiciously short maps.
    if len(notes) < 50:
        raise ProcessingError("Too few notes")

    if not all(validate_note(n) for n in notes):
        raise ProcessingError("Invalid note")

    return None


def validate_note(note):
    keys = ("color", "column", "row", "direction")
    return all(isinstance(note[k], int) for k in keys) and (
        0 <= note["color"] <= 1
        and 0 <= note["direction"] <= 9
        and 0 <= note["column"] <= 3
        and 0 <= note["row"] <= 2
    )
