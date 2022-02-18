import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from config import HogConfig
from dataset import PigPenDataModule
from models.model import Piglet

cs = ConfigStore.instance()
cs.store(name="base_config", node=HogConfig)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: HogConfig)->None:
    print(OmegaConf.to_yaml(cfg))
    
    seed_everything(cfg.train.seed)

    wandb_logger = WandbLogger(
        project="hog",
        entity="itl",
        job_type=cfg.train.job_type,
        config=cfg,
        save_dir=f"{cfg.paths.output_dir}",
    )

    run_name = wandb_logger.experiment.name

    print("Loading dataset")
    pigpen = PigPenDataModule(
        data_dir=cfg.paths.input_dir, batch_size=cfg.train.batch_size, images=cfg.train.images
    )

    print("Creating Model")
    model = Piglet(
        reverse_object_mapping_dir=cfg.paths.input_dir,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        encode_images=cfg.train.images,
    )

    print("Creating Trainer")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", dirpath=f"{cfg.paths.output_dir}/checkpoints/{run_name}"
    )
    if cfg.train.fast:
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            logger=wandb_logger,
            gpus=cfg.train.gpus,
            callbacks=[checkpoint_callback],
            limit_train_batches=10,
            limit_val_batches=10,
            log_every_n_steps=1,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            logger=wandb_logger,
            gpus=cfg.train.gpus,
            callbacks=[checkpoint_callback],
            val_check_interval=0.1,  # check val 10x per epoch
        )

    print("Training...")
    trainer.fit(model, datamodule=pigpen)

    print("Loading Annotated Dataset")
    # pigpen_annotated = PigPenDataModule(
    #     data_dir=f"{hparams.input_dir}/annotated",
    #     batch_size=hparams.batch_size,
    #     annotations=True,
    # )

    # annotations_model = Piglet.load_from_checkpoint(
    #     checkpoint_callback.best_model_path,
    #     strict=False,
    #     symbolic_action=False,
    #     learning_rate=0.00001,
    # )


if __name__ == "__main__":
    main()
    