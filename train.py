import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from peft import (
    LoftQConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from rich import print as rprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from callbacks import AudioLoggingCallback
from config import BeatSaberConfig
from model.model import BeatSaberModel
from optimizer import create_optimizer

os.environ["WANDB_PROJECT"] = "beat_saber"
os.environ["WANDB_LOG_MODEL"] = "false"

# Use tf32 rather than fp32 when using PEFT, which upcasts layer norms.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class BeatSaberTrainingArguments(TrainingArguments):
    learning_rate: float = field(default=1e-6)
    use_conformer: bool = field(default=False)
    conformer_dim: int = field(default=256)
    conformer_depth: int = field(default=8)
    conformer_heads: int = field(default=4)
    conformer_dim_head: int = field(default=64)
    perceiver_dim: int = field(default=256)
    perceiver_depth: int = field(default=8)
    perceiver_heads: int = field(default=4)
    perceiver_dim_head: int = field(default=64)

    learning_rate_lm: float = field(default=5e-6)
    learning_rate_embedding: float = field(default=1e-4)
    loraplus_lr_ratio: float = field(default=16.0)
    rank: int = field(default=128)
    use_lora: bool = field(default=True)
    use_schedulefree_optim: bool = field(default=False)


class BeatSaberTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(BeatSaberTrainer, self).__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_optimizer(self.model, self.args)
        return super().create_optimizer()


@dataclass
class BeatSaberCollator:
    def __call__(self, samples):
        collated_samples = []
        for sample in samples:
            collated_samples.append(
                {
                    "header": sample["retokenized_header"],
                    "segments": sample["retokenized_segments"],
                    "codec_embeddings": sample["codec_embeddings"],
                }
            )
        return {"samples": collated_samples}


def hp_space(trial):
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "use_conformer": trial.suggest_categorical("use_conformer", [True, False]),
        "perceiver_dim": trial.suggest_categorical("perceiver_dim", [64, 256]),
        "perceiver_depth": trial.suggest_int("perceiver_depth", 2, 16),
        "perceiver_heads": trial.suggest_categorical("perceiver_heads", [4, 8, 32]),
        "perceiver_dim_head": trial.suggest_categorical("perceiver_dim_head", [16, 64, 128]),
    }

    if hp["use_conformer"]:
        hp.update(
            {
                "conformer_dim": trial.suggest_categorical("conformer_dim", [64, 256, 512]),
                "conformer_depth": trial.suggest_int("conformer_depth", 2, 16),
                "conformer_heads": trial.suggest_categorical("conformer_heads", [4, 8, 32]),
                "conformer_dim_head": trial.suggest_categorical(
                    "conformer_dim_head", [16, 64, 128]
                ),
            }
        )

    return hp


# def hp_space(trial):
#     hp = {
#         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
#         "use_conformer": trial.suggest_categorical("use_conformer", [True, False]),
#         "perceiver_dim": trial.suggest_categorical("perceiver_dim", [64, 256]),
#         "perceiver_depth": trial.suggest_int("perceiver_depth", 4, 12),
#         "perceiver_heads": trial.suggest_categorical("perceiver_heads", [8, 64]),
#         "perceiver_dim_head": trial.suggest_categorical("perceiver_dim_head", [32, 128]),
#     }

#     if hp["use_conformer"]:
#         hp.update(
#             {
#                 "conformer_dim": trial.suggest_categorical("conformer_dim", [64, 256]),
#                 "conformer_depth": trial.suggest_int("conformer_depth", 2, 16),
#                 "conformer_heads": trial.suggest_categorical("conformer_heads", [8, 32]),
#                 "conformer_dim_head": trial.suggest_categorical(
#                     "conformer_dim_head", [32, 128]
#                 ),
#             }
#         )

#     return hp


def apply_hyperparameters(config, params):
    config.model.learning_rate = params["learning_rate"]
    config.model.use_conformer = params["use_conformer"]

    if config.model.use_conformer:
        config.conformer.dim = params["conformer_dim"]
        config.conformer.depth = params["conformer_depth"]
        config.conformer.heads = params["conformer_heads"]
        config.conformer.dim_head = params["conformer_dim_head"]

    config.perceiver.dim = params["perceiver_dim"]
    config.perceiver.depth = params["perceiver_depth"]
    config.perceiver.heads = params["perceiver_heads"]
    config.perceiver.dim_head = params["perceiver_dim_head"]


def model_init(trial=None):
    config = BeatSaberConfig()
    with torch.device("cuda"):
        if trial:
            params = hp_space(trial)
            apply_hyperparameters(config, params)
            rprint("Trial:", trial.params)

        tokenizer = AutoTokenizer.from_pretrained(config.lm.model)

        if config.lm.use_lora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.lm.use_qlora,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model_args = {}
            if config.lm.use_qlora:
                model_args["quantization_config"] = quantization_config
            else:
                model_args["torch_dtype"] = torch.bfloat16

            lm = AutoModelForCausalLM.from_pretrained(
                config.lm.model,
                attn_implementation="sdpa",
                token=os.environ.get("HF_TOKEN"),
                **model_args,
            )

            if config.lm.use_qlora:
                lm = prepare_model_for_kbit_training(lm)

            lora_args = {}
            if config.lm.use_loftq:
                lora_args["loftq_config"] = LoftQConfig(loftq_bits=4, loftq_iter=1)

            lora_config = LoraConfig(
                r=config.lm.rank,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                use_rslora=True,
                target_modules="all-linear",
                **lora_args,
            )
            lm = get_peft_model(lm, lora_config)
        else:
            lm = AutoModelForCausalLM.from_pretrained(
                config.lm.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                token=os.environ.get("HF_TOKEN"),
            )

        return BeatSaberModel(config, lm, tokenizer)


def load_dataset(config, data_path):
    shard_dirs = (Path(data_path) / "postprocessed").iterdir()
    shards = [load_from_disk(str(p)) for p in shard_dirs]
    dataset = concatenate_datasets(shards).select(range(5000))
    return dataset.train_test_split(test_size=config.model.test_size, shuffle=True, seed=42)


if __name__ == "__main__":
    config = BeatSaberConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.lm.model)

    if len(sys.argv) < 3:
        rprint("Usage: python train.py <data_path> <output_path> [run_name]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2]
    run_name = sys.argv[3] if len(sys.argv) > 3 else None

    train_args = BeatSaberTrainingArguments(
        per_device_train_batch_size=config.model.batch_size,
        per_device_eval_batch_size=config.model.batch_size,
        gradient_accumulation_steps=config.model.gradient_accumulation_steps,
        num_train_epochs=config.model.num_epochs,
        lr_scheduler_type="cosine",
        learning_rate=config.model.learning_rate,
        learning_rate_lm=config.model.learning_rate_lm,
        loraplus_lr_ratio=config.lm.loraplus_lr_ratio,
        optim="adamw_8bit" if config.lm.use_qlora else "adamw_torch_fused",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        output_dir=output_path,
        run_name=run_name,
        save_strategy="steps",
        save_steps=config.model.eval_steps,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=config.model.eval_steps,
        report_to="wandb",
        logging_steps=1,
    )

    dataset_splits = load_dataset(config, data_path)

    trainer = BeatSaberTrainer(
        model_init=model_init,
        args=train_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        tokenizer=tokenizer,
        data_collator=BeatSaberCollator(),
        callbacks=[AudioLoggingCallback()],
    )

    # Uncomment for hyperparameter search:
    # best_trials = trainer.hyperparameter_search(
    #     direction="minimize",
    #     backend="optuna",
    #     hp_space=hp_space,
    #     n_trials=25,
    #     compute_objective=lambda metrics: metrics["eval_loss"],
    # )
    # rprint("Best trial hyperparameters:", best_trials[0])

    trainer.train()
