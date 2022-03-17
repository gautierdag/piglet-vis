import os

import hydra
import pytorch_lightning as pl
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.seed import seed_everything

from config import HogConfig
from dataset import PigPenDataModule
from models.model import Piglet

cs = ConfigStore.instance()
cs.store(name="base_config", node=HogConfig)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: HogConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        name=f"{cfg.run_name}_{cfg.seed}",
        project="hog",
        entity="itl",
        job_type=cfg.pretrain.job_type,
        config=cfg,
        save_dir=f"{cfg.paths.output_dir}",
        mode="disabled" if cfg.fast else "online",
        id=f"{cfg.run_name}_{cfg.seed}",
        group=cfg.run_name,
    )

    run_name = wandb_logger.experiment.name

    print("Loading dataset")
    pigpen = PigPenDataModule(
        data_dir_path=cfg.paths.input_dir,
        batch_size=cfg.pretrain.batch_size,
        images=cfg.images,
        num_workers=cfg.num_workers,
    )

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
    )

    print("Creating Trainer")
    checkpoint_path = f"{cfg.paths.output_dir}/checkpoints/{run_name}"
    best_model_name = f"{cfg.seed}_best"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        every_n_epochs=1,
        filename=best_model_name,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.pretrain.max_epochs,
        logger=wandb_logger,
        gpus=cfg.gpus,
        callbacks=[checkpoint_callback],
        val_check_interval=0.2,  # check val 5x per epoch
        fast_dev_run=cfg.fast,
        strategy=DDPPlugin(find_unused_parameters=False),
    )

    print("Training...")
    if os.path.exists(f"{checkpoint_path}/{best_model_name}.ckpt"):
        print("Found checkpoint: resuming training for model")
        trainer.fit(
            model,
            datamodule=pigpen,
            ckpt_path=f"{checkpoint_path}/{best_model_name}.ckpt",
        )
    else:
        trainer.fit(model, datamodule=pigpen)

    wandb.finish()

    print("Loading NLU Task..")
    run_name = f"{cfg.run_name}_nlu_task"
    pigpen = PigPenDataModule(
        data_dir_path=f"{cfg.paths.input_dir}/annotated",
        output_dir_path=f"{cfg.paths.output_dir}",
        batch_size=cfg.nlu.batch_size,
        annotations=True,
        num_workers=cfg.num_workers,
    )

    model = Piglet.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        strict=False,
        symbolic_action=False,
        learning_rate=cfg.nlu.learning_rate,
        output_dir_path=f"{cfg.paths.output_dir}",
    )

    checkpoint_path = f"{cfg.paths.output_dir}/checkpoints/{run_name}"
    best_model_name = f"{cfg.seed}_best"
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        every_n_epochs=1,
        filename=best_model_name,
    )

    wandb_logger = WandbLogger(
        name=f"{cfg.run_name}_{cfg.seed}",
        project="hog",
        entity="itl",
        job_type=cfg.nlu.job_type,
        config=cfg,
        save_dir=f"{cfg.paths.output_dir}",
        mode="disabled" if cfg.fast else "enabled",
        id=f"{cfg.run_name}_{cfg.seed}",
        group=cfg.run_name,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.nlu.max_epochs,
        logger=wandb_logger,
        gpus=cfg.gpus,
        callbacks=[checkpoint_callback],
        fast_dev_run=cfg.fast,
        log_every_n_steps=1,
        strategy=DDPPlugin(find_unused_parameters=False),
        fuse_images=cfg.model.fuse_images,
    )

    print("Training...")
    if os.path.exists(f"{checkpoint_path}/{best_model_name}.ckpt"):
        print("Found checkpoint: resuming training for model")
        trainer.fit(
            model,
            datamodule=pigpen,
            ckpt_path=f"{checkpoint_path}/{best_model_name}.ckpt",
        )
    else:
        trainer.fit(model, datamodule=pigpen)

    print("Testing...")
    trainer.test(model, datamodule=pigpen)


if __name__ == "__main__":
    main()
