from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:
    input_dir: str
    output_dir: str


@dataclass
class Model:
    hidden_size: int
    num_layers: int
    num_heads: int
    bert_model: str
    dropout: float
    fuse_images: bool
    no_symbolic: bool
    label_name_embeddings: bool


@dataclass
class Train:
    batch_size: int
    max_epochs: int
    learning_rate: float
    checkpoint_path: Optional[str] = None


@dataclass
class HogConfig:
    run_name: str
    seed: int
    gpus: str  # use "1" to use only one gpu
    fast: bool  # whether to run fast dev run
    images: bool  # whether to use images
    num_workers: int  # number of workers for dataloader
    paths: Paths  # input and output directories
    model: Model  # model parameters constant for both pretrain and nlu
    pretrain: Train  # Settings specific to pretraining
    nlu: Train  # Settings specific to NLU finetuning task
