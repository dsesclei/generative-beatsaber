### Audio encoder + LLM for rhythm game map generation

During preprocessing, songs are divided into segments roughly 10 seconds long, containing N beats each (varies per song by BPM.) The raw audio for these segment is converted into embeddings with [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).

In training, these embeddings are passed through a conformer followed by a Q-former (BLIP-2), which converts this variable length segment into a fixed number of LLM embeddings to be used in place of tokens.

For each sample, the LLM is shown all the audio embeddings, a header, and then the audio embeddings for each segment interleaved with the segment note tokens. Phrased in code:

```
all_audio_embeddings + header_tokens + audio_embeddings[0] + segment_tokens[0] + audio_embeddings[1] + segment_tokens[1] ...
```

Or in tokens:

```
AUDIO_0 AUDIO_1 ... AUDIO_N <header> Difficulty: expert-plus | BPM level: 3 | Rating: 9 | walls </header> AUDIO_0 [red middle far-left down] [blue bottom left down-left] [12% blue bottom far-right right] [25% blue bottom left left] ... end AUDIO_1 start [red bottom right right 12%] [red middle far-left up-left] ...
```

Notes are in the format `[percent along segment, color, row, col, cut direction]`.

Code is also in place for experimenting with spectrograms instead of the codec, and a Perceiver (BLIP-3) rather than the Q-former.
