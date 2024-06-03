"""
Optimizer with LoRA+ and different param group for the new layers.

LoRA+ scales the LR of the B matrix to improve performance: https://x.com/hayou_soufiane/status/1760033513800450486

Adapted from LLaMA-Factory:
https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llmtuner/train/utils.py
"""

from typing import Dict, List

import torch
from rich import print as rprint
from schedulefree import AdamWScheduleFree
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


def _get_decay_parameter_names(model):
    """
    Return parameters with weight decay (weights in non-layernorm layers.)
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def create_optimizer(model, training_args):
    default_lr = training_args.learning_rate
    lm_lr = training_args.learning_rate_lm
    loraplus_lr = lm_lr * training_args.loraplus_lr_ratio
    embedding_lr = training_args.learning_rate_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: Dict[str, List["torch.nn.Parameter"]] = {
        "audio": [],
        "lm": [],
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "embed_tokens" in name or ".lm_head." in name:
                param_dict["embedding"].append(param)
            elif model.config.lm.use_lora and (".lora_B." in name or param.ndim == 1):
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            elif model.config.lm.use_lora and ".lora_A." in name:
                param_dict["lora_a"].append(param)
            elif name.startswith("audio_embedder.") or name == "query_tokens":
                param_dict["audio"].append(param)
            elif name.startswith("lm."):
                param_dict["lm"].append(param)
            else:
                rprint(f"Unrecognized parameter: {name}")

    for key, val in param_dict.items():
        rprint(f"Group {key} has {len(val)} parameters")

    if training_args.use_schedulefree_optim:
        _, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
        optim_kwargs.update({"r": 0.0, "weight_lr_power": 2.0})
        optim_class = AdamWScheduleFree
    else:
        optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        dict(
            params=param_dict["audio"],
            lr=default_lr,
            weight_decay=training_args.weight_decay,
        ),
        dict(
            params=param_dict["lm"],
            lr=lm_lr,
            weight_decay=training_args.weight_decay,
        ),
        dict(
            params=param_dict["lora_a"],
            lr=default_lr,
            weight_decay=training_args.weight_decay,
        ),
        dict(
            params=param_dict["lora_b"],
            lr=loraplus_lr,
            weight_decay=training_args.weight_decay,
        ),
        dict(params=param_dict["lora_b_nodecay"], lr=loraplus_lr, weight_decay=0.0),
        dict(
            params=param_dict["embedding"],
            lr=embedding_lr,
            weight_decay=training_args.weight_decay,
        ),
    ]

    optimizer = optim_class(param_groups, **optim_kwargs)
    return optimizer
