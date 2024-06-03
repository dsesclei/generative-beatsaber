import torch
import torch.nn as nn
from conformer import Conformer
from transformers import Blip2QFormerConfig, Blip2QFormerModel

from .perceiver import AudioPerceiver


class AudioEmbedder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.conformer_processor = (
            ConformerProcessor(config) if config.model.use_conformer else None
        )
        self.embedder = self._initialize_embedder(config).to(torch.bfloat16)
        self.project_input = nn.Linear(
            config.data.mel_bands
            if config.model.audio_key == "spectrograms"
            else config.data.codec_dim,
            config.conformer.dim
            if config.model.use_conformer
            else (
                config.qformer.dim if config.model.embedder == "qformer" else config.perceiver.dim
            ),
        ).to(torch.bfloat16)
        self.to(device).to(torch.bfloat16)

    def _initialize_embedder(self, config):
        if config.model.embedder == "qformer":
            return QFormerEmbedder(config)
        else:
            return PerceiverEmbedder(config)

    def embed_audio(self, audio_inputs, batch_size=256):
        outputs = []
        for i in range(0, len(audio_inputs), batch_size):
            batch = torch.tensor(
                audio_inputs[i : i + batch_size],
                dtype=torch.bfloat16,
                device=self.device,
            )

            if self.config.model.audio_key == "codec_embeddings":
                batch = batch.permute(0, 2, 1)

            projected_batch = self.project_input(batch)
            outputs.append(self.embedder.embed(projected_batch, self.conformer_processor))
        return torch.cat(outputs, dim=0)


class ConformerProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conformer = Conformer(
            dim=config.conformer.dim,
            depth=config.conformer.depth,
            dim_head=config.conformer.dim_head,
            heads=config.conformer.heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1,
        )
        self.project_conformer = nn.Linear(
            config.conformer.dim,
            config.qformer.dim if config.model.embedder == "qformer" else config.perceiver.dim,
        )

    def process(self, inputs):
        conformer_output = self.conformer(inputs)
        return self.project_conformer(conformer_output)


class QFormerEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qformer = Blip2QFormerModel(
            Blip2QFormerConfig(
                hidden_size=config.qformer.dim,
                num_hidden_layers=config.qformer.layers,
                num_attention_heads=config.qformer.heads,
                encoder_hidden_size=config.qformer.dim,
                intermediate_size=config.qformer.dim * 4,
            )
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.model.num_audio_tokens, config.qformer.dim)
        )
        self.project_qformer = nn.Linear(config.qformer.dim, config.lm.dim)

    def embed(self, batch, conformer):
        if conformer:
            processed_batch = conformer.process(batch)
        else:
            processed_batch = batch

        qformer_out = self.qformer(
            query_embeds=self.query_tokens.expand(processed_batch.size(0), -1, -1),
            encoder_hidden_states=processed_batch,
        )
        return self.project_qformer(qformer_out.last_hidden_state)


class PerceiverEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.perceiver = AudioPerceiver(
            dim=config.perceiver.dim,
            depth=config.perceiver.depth,
            heads=config.perceiver.heads,
            dim_head=config.perceiver.dim_head,
            num_latents=config.model.num_audio_tokens,
            max_time_steps=1000,
        )
        self.project_perceiver = nn.Linear(config.perceiver.dim, config.lm.dim)

    def embed(self, batch, conformer):
        if conformer:
            processed_batch = conformer.process(batch)
        else:
            processed_batch = batch

        latents = self.perceiver(processed_batch)
        projected_latents = self.project_perceiver(latents)
        return projected_latents
