### Audio encoder + LLM for rhythm game map generation

### Architecture
![diagram](https://github.com/dsesclei/generative-beatsaber/assets/801452/555d73f5-11c1-4f98-a762-7840e777aa06)

#### Preprocessing
Songs are divided into segments roughly 10 seconds long, each containing a number of beats (which varies per song by BPM.) The raw audio for these songs is then processed through a neural audio codec ([Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)), providing the initial audio embeddings.

#### Training
These codec embeddings are further processed by a Conformer followed by a Perceiver (BLIP-3), the latter of which converts this variable length segment into a fixed number of LLM embeddings that can be used in place of tokens.

Llama-3-8b is finetuned on these songs with their tokenized beatmaps using LoRA. The prompt is composed of the full list of audio embeddings, a header, and then the embeddings for each segment interleaved with the segment's note tokens. Phrased in code:

```
all_audio_embeddings + header_tokens + audio_embeddings[0] + segment_tokens[0] + audio_embeddings[1] + segment_tokens[1] ...
```

Or in tokens:

```
AUDIO_0 AUDIO_1 ... AUDIO_N <header> Difficulty: expert-plus | BPM level: 3 | Rating: 9 | walls </header> AUDIO_0 [red middle far-left down] [blue bottom left down-left] [12% blue bottom far-right right] [25% blue bottom left left] ... end AUDIO_1 start [red bottom right right] [12% red middle far-left up-left] ...
```

Notes are in the format `[percent along segment, color, row, col, cut direction]`.

Code is also present for experimenting with spectrograms instead of the codec, and a Q-former (BLIP-2) in place of the Perceiver.

#### Similar projects and papers

- [BeatLearning](https://github.com/sedthh/BeatLearning) - for OSU, also transformer based
- [InfernoSaber](https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper), which separates responsibilites out into individual convolutional/FFN models
- [Beat Sage](https://beatsage.com/), based on the paper [Dance Dance Convolution](https://arxiv.org/abs/1703.06891)
- [An Embarrassingly Simple Approach for LLM with Strong ASR Capacity](https://arxiv.org/abs/2402.08846) - helpful overview of recent audio+LLM papers in Table 1
- [Connecting Speech Encoder and Large Language Model for ASR](https://arxiv.org/abs/2309.13963) - frozen encoder + trainable Q-former + frozen LLM for ASR
