from dataclasses import dataclass


@dataclass
class Paths:
    input_dir: str
    output_dir: str


@dataclass
class Train:
    gpus: str
    job_type: str
    batch_size: int
    max_epochs: int
    seed: int
    fast: bool
    images: bool


@dataclass
class Model:
    hidden_size: int
    num_layers: int
    num_heads: int
    bert_model: str
    dropout: float


@dataclass
class HogConfig:
    paths: Paths
    train: Train
    model: Model
