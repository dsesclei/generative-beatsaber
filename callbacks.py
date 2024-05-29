import math

import wandb
from transformers import TrainerCallback


class AudioLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model, **kwargs):
        logs = {}

        l1_norm_sum = 0.0
        l2_norm_sum_sq = 0.0
        for p in model.audio_embedder.parameters():
            if p.grad is not None:
                l1_norm_sum += p.grad.detach().abs().sum().item()
                l2_norm_sum_sq += p.grad.detach().pow(2).sum().item()
        if l1_norm_sum > 0 or l2_norm_sum_sq > 0:
            total_l2_norm = math.sqrt(l2_norm_sum_sq)
            logs |= {
                "grad_norm_l1": l1_norm_sum,
                "grad_norm_l2": total_l2_norm,
            }
        else:
            print("No gradients found for audio embedder at this step")

        if model.loggable_embeds is not None:
            logs |= {
                # "embeds": model.loggable_embeds,
                "embeds_mean": model.loggable_embeds.mean(),
                "embeds_variance": model.loggable_embeds.var(),
            }

        wandb.log({"AudioEmbedder": logs})
        return control
