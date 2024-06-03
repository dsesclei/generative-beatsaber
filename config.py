from dataclasses import asdict, dataclass, field
from transformers import PretrainedConfig


@dataclass
class DataConfig:
    segment_length: float = 9.6
    max_songs: int = 2000
    shard_size: int = 100
    mel_bands: int = 80
    codec_dim: int = 9
    only_highest_difficulty: bool = True
    preprocess_workers: int = 32
    codec_batch_size: int = 4
    # Token format.
    note_format: str = "[NOTE_{color}_{row}_{column}_{direction}]"
    num_time_tokens: int = 100


@dataclass
class ModelConfig:
    test_size: float = 0.1
    batch_size: int = 1  # Use 1 for Llama 3, which doesn't have a pad token by default.
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    eval_steps: int = 2500
    learning_rate: float = 5e-4
    learning_rate_lm: float = 2e-5
    learning_rate_embedding: float = 2e-5
    use_schedulefree_optim: bool = False
    num_audio_tokens: int = 8
    embedder: str = "perceiver"
    audio_key: str = "codec_embeddings"
    use_conformer: bool = True


@dataclass
class LMConfig:
    # model: str = "unsloth/gemma-2b"
    # dim: int = 2048
    freeze: bool = False  # Set use_lora to false if true.
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dim: int = 4096
    context_size: int = 8192
    # LoRA config.
    loraplus_lr_ratio: float = 16.0
    rank: int = 32
    use_lora: bool = True
    use_qlora: bool = False
    use_loftq: bool = False


@dataclass
class QFormerConfig:
    dim: int = 128
    layers: int = 2
    heads: int = 4


@dataclass
class PerceiverConfig:
    depth: int = 4
    dim: int = 64
    dim_head: int = 32
    heads: int = 16


@dataclass
class ConformerConfig:
    depth: int = 4
    dim: int = 128
    dim_head: int = 64
    heads: int = 8


@dataclass
class BeatSaberConfig(PretrainedConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    qformer: QFormerConfig = field(default_factory=QFormerConfig)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get("data", DataConfig())
        self.model = kwargs.get("model", ModelConfig())
        self.lm = kwargs.get("lm", LMConfig())
        self.qformer = kwargs.get("qformer", QFormerConfig())
        self.perceiver = kwargs.get("perceiver", PerceiverConfig())
        self.conformer = kwargs.get("conformer", ConformerConfig())
        self.model_type = "beatsaber"

    def to_dict(self):
        output = super().to_dict()
        output.update(
            {
                "data": asdict(self.data),
                "model": asdict(self.model),
                "lm": asdict(self.lm),
                "qformer": asdict(self.qformer),
                "perceiver": asdict(self.perceiver),
                "conformer": asdict(self.conformer),
            }
        )
        return output

    @classmethod
    def from_dict(cls, config_dict):
        data = DataConfig(**config_dict.pop("data"))
        model = ModelConfig(**config_dict.pop("model"))
        lm = LMConfig(**config_dict.pop("lm"))
        qformer = QFormerConfig(**config_dict.pop("qformer"))
        perceiver = PerceiverConfig(**config_dict.pop("perceiver"))
        conformer = ConformerConfig(**config_dict.pop("conformer"))
        config = cls(
            data=data,
            model=model,
            lm=lm,
            qformer=qformer,
            perceiver=perceiver,
            conformer=conformer,
            **config_dict
        )
        return config
