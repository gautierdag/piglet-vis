import glob
import os
from typing import Tuple

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from config import HogConfig
from dataset import PigPenDataModule
from models.model import Piglet


def get_unique_run_name(checkpoint_path: str, seed: int) -> Tuple[str, bool]:
    # if directory exists and there is only one checkpoint
    if os.path.exists(checkpoint_path):
        checkpoints = glob.glob(f"{checkpoint_path}/*.ckpt")
        if len(checkpoints) == 1:
            return checkpoints[0].split(".ckpt")[0], True
    # else return get_unique_run_name, False (not resuming)
    unique_run_name = f"{seed}_{wandb.util.generate_id()}"
    return unique_run_name, False


def train(cfg: HogConfig, job_type="pretrain", best_model_path=None) -> str:
    """
    Train the model for different modes (pretrain/nlu).
    :param cfg: Config object
    :param job_type: pretrain or nlu
    :param best_model_path: path to the best model

    :return: path to the best model
    """
    # Determine name of run and unique id (if not resuming)
    run_name = f"{cfg.run_name}_{cfg.seed}"

    if job_type == "pretrain":
        data_dir_path = cfg.paths.input_dir
        annotations = False
        pretrain = True
    elif job_type == "nlu":
        assert best_model_path is not None, "Must provide best_model_path for nlu"
        run_name += "_nlu"
        data_dir_path = f"{cfg.paths.input_dir}/annotated"
        annotations = True
        pretrain = False
    else:
        raise NotImplementedError(f"job_type {job_type} not implemented")

    checkpoint_path = f"{cfg.paths.output_dir}/checkpoints/{run_name}"
    unique_run_name, resume_training = get_unique_run_name(checkpoint_path, cfg.seed)

    wandb_logger = WandbLogger(
        name=run_name,
        project="hog",
        entity="itl",
        job_type=job_type,
        config=cfg,
        save_dir=f"{cfg.paths.output_dir}",
        mode="disabled" if cfg.fast else "online",
        id=unique_run_name,
        group=cfg.run_name,
    )

    print("Loading dataset")
    pigpen = PigPenDataModule(
        data_dir_path=data_dir_path,
        batch_size=cfg[job_type].batch_size,
        images=cfg.images,
        annotations=annotations,
        num_workers=cfg.num_workers,
    )

    if best_model_path:
        print("Loading Model from checkpoint")
        model = Piglet.load_from_checkpoint(
            best_model_path,
            strict=False,
            pretrain=pretrain,
            learning_rate=cfg[job_type].learning_rate,
            output_dir_path=f"{cfg.paths.output_dir}",
        )
    else:
        print("Creating Model")
        model = Piglet(
            data_dir_path=cfg.paths.input_dir,
            output_dir_path=f"{cfg.paths.output_dir}",
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            encode_images=cfg.images,
            fuse_images=cfg.model.fuse_images,
            learning_rate=cfg[job_type].learning_rate,
            pretrain=pretrain,
        )

    print("Creating Trainer")
    # best_model_name = f"{cfg.seed}_best"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        every_n_epochs=1,
        filename=unique_run_name,
    )

    trainer = pl.Trainer(
        max_epochs=cfg[job_type].max_epochs,
        logger=wandb_logger,
        gpus=cfg.gpus,
        callbacks=[checkpoint_callback],
        val_check_interval=0.2,  # check val 5x per epoch
        fast_dev_run=cfg.fast,
        strategy=DDPPlugin(find_unused_parameters=False),
    )

    print("Training...")
    if resume_training:
        print("Found checkpoint: resuming training for model")
        trainer.fit(
            model,
            datamodule=pigpen,
            ckpt_path=f"{checkpoint_path}/{unique_run_name}.ckpt",
        )
    else:
        trainer.fit(model, datamodule=pigpen)

    if job_type == "nlu":
        print("Testing...")
        trainer.test(model, datamodule=pigpen)

    wandb.finish()

    return checkpoint_callback.best_model_path
