"""
Script to adapt model to Beat Saber structure before training.
Trains on dropped samples to create a model we can further finetune with a frozen LM.
Not sure if this would help - it might cause the model to rely too much on its knowledge of chart structure.
"""
import os
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from config import BeatSaberConfig

os.environ["WANDB_PROJECT"] = "beat_saber_adapt"
os.environ["WANDB_LOG_MODEL"] = "false"


def load_dataset(config, data_path):
    shard_dirs = (Path(data_path) / "dropped").iterdir()
    shards = [load_from_disk(str(p)) for p in shard_dirs]
    dataset = concatenate_datasets(shards).select(range(1000))
    return dataset.train_test_split(test_size=config.model.test_size, shuffle=True, seed=42)


if __name__ == "__main__":
    config = BeatSaberConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.lm.model)
    dataset = load_dataset(config, "/data")

    model = AutoModelForCausalLM.from_pretrained(
        config.lm.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    args = TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        output_dir="/model",
        report_to="wandb",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1000,
        bf16=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        max_steps=1000,
    )

    trainer = SFTTrainer(
        model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=config.lm.context_size,
        packing=True,
    )

    trainer.train()
