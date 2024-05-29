"""
From xgen-mm/BLIP-3:
https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1/blob/main/vlm.py

Unverified(!) adaption for audio by ChatGPT(!). But it seems to learn.
"""
import torch
from einops import rearrange, repeat
from torch import einsum, nn


class AudioPerceiver(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, num_latents, max_time_steps=None):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.time_step_embeddings = (
            nn.Parameter(torch.randn(max_time_steps, dim)) if max_time_steps else None
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        if self.time_step_embeddings is not None:
            max_t = self.time_step_embeddings.shape[0]
            if x.size(1) > max_t:
                raise ValueError(
                    f"Input time dimension {x.size(1)} exceeds maximum allowed {max_t}"
                )
            time_embs = self.time_step_embeddings[: x.size(1)]
            x += time_embs.unsqueeze(0)

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return latents


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, audio_attn_masks=None):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        d = self.dim_head

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=1)

        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=h)
        k = rearrange(k, "b t (h d) -> b h t d", h=h)
        v = rearrange(v, "b t (h d) -> b h t d", h=h)

        q = q * self.scale

        # Corrected einsum subscripts
        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        if audio_attn_masks is not None:
            audio_attn_masks = rearrange(audio_attn_masks, "b t -> b 1 t 1")
            sim = torch.where(audio_attn_masks, sim, float("-inf"))

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.to_out(out)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
