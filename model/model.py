import os

import torch
from config import BeatSaberConfig
from peft import AutoPeftModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .audio_embedder import AudioEmbedder


class BeatSaberModel(PreTrainedModel):
    config_class = BeatSaberConfig
    supports_gradient_checkpointing = True

    def __init__(self, config, lm, tokenizer):
        super().__init__(config)
        self.config = config
        self.lm = lm
        self.tokenizer = tokenizer
        self.audio_embedder = AudioEmbedder(config, self.device)

        self.start_token_id = self.tokenizer.convert_tokens_to_ids("start")
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("end")

        token_tensor = lambda x: torch.tensor([x], dtype=torch.long, device=self.device).view(1, -1)
        self.bos_token = token_tensor(self.tokenizer.bos_token_id)
        self.eos_token = token_tensor(self.tokenizer.eos_token_id)
        self.start_token = token_tensor(self.start_token_id)
        self.end_token = token_tensor(self.end_token_id)

        print(
            f"Total parameters: {self.count_parameters(self)}",
            f"Audio Embedder: {self.count_parameters(self.audio_embedder)}",
        )

        if self.config.lm.freeze:
            for param in self.lm.parameters():
                param.requires_grad = False

        self.loggable_embeds = None
        self.inputs_log = None

    @classmethod
    def from_pretrained(cls, save_directory, *args, **kwargs):
        config = BeatSaberConfig.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)

        if config.lm.use_lora:
            lm = AutoPeftModelForCausalLM.from_pretrained(os.path.join(save_directory, "lm"))
        else:
            lm = AutoModelForCausalLM.from_pretrained(
                os.path.join(save_directory, "lm"),
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )

        model = cls(config, lm, tokenizer, *args, **kwargs)

        audio_embedder_state_dict = torch.load(os.path.join(save_directory, "audio_embedder.pth"))
        model.audio_embedder.load_state_dict(audio_embedder_state_dict)

        return model

    def save_pretrained(self, save_directory, **kwargs):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.config.save_pretrained(save_directory)
        self.lm.save_pretrained(os.path.join(save_directory, "lm"))
        torch.save(
            self.audio_embedder.state_dict(),
            os.path.join(save_directory, "audio_embedder.pth"),
        )
        self.tokenizer.save_pretrained(save_directory)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _token_embedding(self, token_id):
        return self.lm.get_input_embeddings()(token_id).view(1, self.config.lm.dim)

    def _prepare_header(self, embeds, labels, audio_embeds, token_ids):
        """
        Header format:
        bos, audio_0, ..., audio_N, token_0, ..., token_N
        """
        embeds[0] = self._token_embedding(self.bos_token)
        labels[0] = self.bos_token.view(1)
        index = 1

        flat_audio_embeds = audio_embeds.view(-1, self.config.lm.dim)
        embeds[index : index + len(flat_audio_embeds)] = flat_audio_embeds
        labels[index : index + len(flat_audio_embeds)] = torch.full(
            (len(flat_audio_embeds),), -100, dtype=torch.long, device=self.device
        )
        index += len(flat_audio_embeds)

        token_embeds = self.lm.get_input_embeddings()(token_ids.to(self.device))
        embeds[index : index + len(token_embeds)] = token_embeds
        labels[index : index + len(token_ids)] = token_ids
        index += len(token_embeds)

        return index

    def _prepare_segment(self, index, embeds, labels, audio_embeds, segment_ids):
        """
        Segment format: [start, audio_0, ..., audio_N, token_0, ..., token_N, end]
        """
        embeds[index] = self._token_embedding(self.start_token)
        labels[index] = self.start_token.view(1)
        index += 1

        embeds[index : index + len(audio_embeds)] = audio_embeds
        labels[index : index + len(audio_embeds)] = torch.full(
            (len(audio_embeds),), -100, dtype=torch.long, device=self.device
        )
        index += len(audio_embeds)

        segment_embeds = self.lm.get_input_embeddings()(segment_ids)
        embeds[index : index + len(segment_embeds)] = segment_embeds
        labels[index : index + len(segment_embeds)] = segment_ids
        index += len(segment_embeds)

        embeds[index] = self._token_embedding(self.end_token)
        labels[index] = self.end_token.view(1)
        index += 1

        return index

    def _prepare_sample(self, sample, audio_embeds):
        total_audio_length = 2 * len(audio_embeds) * self.config.model.num_audio_tokens
        total_segment_length = sum(1 + len(s) + 1 for s in sample["segments"])
        total_length = 1 + len(sample["header"]) + total_audio_length + total_segment_length + 1

        embeds = torch.zeros(
            (total_length, self.config.lm.dim), device=self.device, dtype=torch.bfloat16
        )
        labels = torch.full((total_length,), -100, dtype=torch.long, device=self.device)

        index = self._prepare_header(embeds, labels, audio_embeds, sample["header"])

        for segment_audio, segment_ids in zip(audio_embeds, sample["segments"]):
            index = self._prepare_segment(index, embeds, labels, segment_audio, segment_ids)

        embeds[index] = self._token_embedding(self.eos_token)
        labels[index] = self.eos_token.view(1)

        mask = torch.full((len(embeds),), 1, dtype=torch.long, device=self.device)

        return embeds, labels, mask

    def forward(self, samples, return_loss=True):
        for sample in samples:
            sample["header"] = torch.tensor(sample["header"], dtype=torch.long, device=self.device)
            sample["segments"] = [
                torch.tensor(segment, dtype=torch.long, device=self.device)
                for segment in sample["segments"]
            ]

        sample_embeds = [
            self.audio_embedder.embed_audio(sample[self.config.model.audio_key])
            for sample in samples
        ]

        if self.training:
            self.loggable_embeds = (
                torch.cat(sample_embeds, dim=0)
                .view(-1, self.config.lm.dim)
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype("float32")
            )

        prepared_samples = [
            self._prepare_sample(sample, audio_embeds)
            for sample, audio_embeds in zip(samples, sample_embeds)
        ]
        embeds_list, labels_list, masks_list = zip(*prepared_samples)

        embeds = pad_sequence(embeds_list, batch_first=True)
        masks = pad_sequence(masks_list, batch_first=True)
        labels = pad_sequence(labels_list, batch_first=True)

        outputs = self.lm(inputs_embeds=embeds, attention_mask=masks, labels=labels)

        return CausalLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=None,
            attentions=None,
        )

    def generate(self, audio_inputs, header_tokens):
        audio_embeds = self.audio_embedder.embed_audio(audio_inputs)
        header_ids = self.tokenizer(header_tokens, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].squeeze(0)
        header_length = 1 + len(header_ids) + len(audio_embeds) * self.config.model.num_audio_tokens
        embeds = torch.zeros(
            (header_length, self.config.lm.dim),
            device=self.device,
            dtype=torch.bfloat16,
        )
        header_labels = torch.full((header_length,), -100, dtype=torch.long, device=self.device)
        _ = self._prepare_header(embeds, header_labels, audio_embeds, header_ids)

        generated_tokens = []
        start_embed = self._token_embedding(self.start_token)
        different = 0
        total = 0
        for segment_audio in audio_embeds:
            embeds = torch.cat([embeds, start_embed, segment_audio], dim=0)
            generated_tokens.append(self.start_token_id)

            while True:
                outputs = self.lm(inputs_embeds=embeds.unsqueeze(0))
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                token_embed = self.lm.get_input_embeddings()(
                    torch.tensor([next_token], device=self.device).view(1)
                )
                embeds = torch.cat([embeds, token_embed], dim=0)
                generated_tokens.append(next_token)

                if next_token == self.end_token:
                    break

        return generated_tokens
