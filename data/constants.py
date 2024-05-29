BEATMAP_KEYS = {
    "3": {
        "top": {
            "notes": "colorNotes",
            "bombs": "bombNotes",
            "walls": "obstacles",
            "sliders": "sliders",
            "burst_sliders": "burstSliders",
        },
        "notes": {
            "beat": "b",
            "color": "c",
            "column": "x",
            "row": "y",
            "direction": "d",
        },
        "bombs": {
            "beat": "b",
        },
        "walls": {
            "beat": "b",
            "duration": "d",
        },
        "sliders": {
            "color": "c",
            "head_beat": "b",
            "head_row": "y",
            "head_column": "x",
            "tail_beat": "tb",
            "tail_row": "ty",
            "tail_column": "tx",
        },
        "burst_sliders": {
            "color": "c",
            "head_beat": "b",
            "head_row": "y",
            "head_column": "x",
            "tail_beat": "tb",
            "tail_row": "ty",
            "tail_column": "tx",
        },
    },
    "2": {
        "top": {
            "notes": "_notes",
            "bombs": "_bombs",
            "walls": "_obstacles",
            "sliders": "_sliders",
        },
        "notes": {
            "beat": "_time",
            "color": "_type",
            "row": "_lineLayer",
            "column": "_lineIndex",
            "direction": "_cutDirection",
        },
        "bombs": {
            "beat": "_time",
        },
        "walls": {
            "beat": "_time",
            "duration": "_duration",
        },
        "sliders": {
            "color": "_colorType",
            "head_beat": "_headTime",
            "head_row": "_headLineLayer",
            "head_column": "_headLineIndex",
            "tail_beat": "_headTime",
            "tail_row": "_headLineLayer",
            "tail_column": "_headLineIndex",
        },
    },
}
