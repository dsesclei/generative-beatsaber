# Example header:
# [DIFFICULTY_EASY] [BPM_1] [RATING_7] [NPS_2] [BOMBS] [WALLS]
def build_header(beatmap):
    quantized_bpm = quantize_bpm(beatmap.bpm)
    quantized_rating = int(beatmap.rating * 10)

    num_beats = beatmap.num_segments * beatmap.beats_per_segment
    npb = min(len(beatmap.notes) / num_beats, 2.5)
    num_npb_bins = 10
    quantized_npb = min(int((npb / (2.5 / num_npb_bins))), num_npb_bins + 1)

    return [
        "[HEADER]",
        f"[DIFFICULTY_{beatmap.difficulty.upper()}]",
        f"[BPM_{quantized_bpm}]",
        f"[RATING_{quantized_rating}]",
        f"[NPB_{quantized_npb}]",
        ## TODO BURST NOTES
        *(["[BOMBS]"] if beatmap.bomb_segments else []),
        *(["[WALLS]"] if beatmap.wall_segments else []),
        "[END_HEADER]",
    ]


# Bucket BPM into bins.
def quantize_bpm(bpm):
    boundaries = [110, 150, 190, 230, 270, 300]

    for index, boundary in enumerate(boundaries, start=1):
        if bpm <= boundary:
            return index

    return len(boundaries) + 1


def build_segments(beatmap, num_time_tokens, note_format):
    segments = []
    note_idx = 0
    for segment_idx in range(beatmap.num_segments):
        segment, note_idx = build_segment(
            beatmap,
            num_time_tokens,
            note_format,
            segment_idx,
            note_idx,
        )
        segments.append(segment)

    return segments


def build_segment(beatmap, num_time_tokens, note_format, segment_idx, note_idx):
    last_time_token = None
    last_time_token_beat = 0

    # Indicate if wall and bomb info is missing from this segment.
    segment = [
        "[SEG]",
        *(["[WALL]"] if segment_idx in beatmap.wall_segments else []),
        *(["[BOMB]"] if segment_idx in beatmap.bomb_segments else []),
    ]

    # Construct segments by walking through the notes.
    while note_idx < len(beatmap.notes):
        note = beatmap.notes[note_idx]
        # If note does not belong to current segment, break and move onto next.
        if note["beat"] // beatmap.beats_per_segment != segment_idx:
            break

        # Time tokens are indicate the relative position of notes within a segment.
        # With 50 time tokens, [TIME_35] indicates that a note is 70% through the segment.
        # Calculate time index for time token.
        segment_pos = note["beat"] - (segment_idx * beatmap.beats_per_segment)
        time_index = get_bin_index(segment_pos, beatmap.beats_per_segment, num_time_tokens)

        # Add token only if note is not at segment start and time has progressed.
        time_token = None
        if time_index != 0:
            time_token = f"[TIME_{time_index}]"
            if time_token != last_time_token:
                last_time_token = time_token
                last_time_token_beat = note["beat"]
                segment.append(time_token)
            elif note["beat"] != last_time_token_beat:
                # When the quantized time delta is too small to represent,
                # add a token indicating a short gap instead.
                segment.append("[SHORT_GAP]")

        # Add note token.
        segment.append(note_format.format(**note))

        # Add slider token.
        if slider_type := beatmap.slider_for_note(note):
            segment.append(f"[SLIDER_{slider_type}_{note['color']}]")

        note_idx += 1

    segment.append("[END_SEG]")

    return segment, note_idx


def get_bin_index(value, length, bin_count):
    bin_index = int((value / length) * bin_count)
    return min(bin_index, bin_count - 1)


# Map BPM to a token: 160 -> [BPM_3]
def bpm_to_token(bpm):
    boundaries = [110, 150, 190, 230, 270, 300]

    for index, boundary in enumerate(boundaries, start=1):
        if bpm <= boundary:
            return f"[BPM_{index}]"

    return f"[BPM_{len(boundaries) + 1}]"
