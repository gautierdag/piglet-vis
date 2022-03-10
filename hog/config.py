from dataclasses import dataclass


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


@dataclass
class Train:
    batch_size: int
    max_epochs: int
    learning_rate: float


@dataclass
class Pretrain(Train):
    job_type: str = "pretrain"


@dataclass
class NLU(Train):
    job_type: str = "nlu_task"


@dataclass
class HogConfig:
    run_name: str
    seed: int
    gpus: str  # use "1" to use only one gpu
    fast: bool  # whether to run fast dev run
    images: bool  # whether to use images
    paths: Paths  # input and output directories
    model: Model  # model parameters constant for both pretrain and nlu
    pretrain: Pretrain  # Settings specific to pretraining
    nlu: NLU  # Settings specific to NLU finetuning task
