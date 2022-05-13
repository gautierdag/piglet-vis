import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from config import HogConfig
from dataset import PigPenDataModule, preprocess_images
from models.model import Piglet


def train(
    cfg: HogConfig,
    job_type="pretrain",
    best_model_path=None,
    resume_from_checkpoint=None,
) -> str:
    """
    Train the model for different modes (pretrain/nlu).
    :param cfg: Config object
    :param job_type: pretrain or nlu
    :param best_model_path: path to the best model
    :param resume_from_checkpoint: checkpoint unique id to resume training from

    :return: path to the best model
    """
    if cfg.images:
        preprocess_images(cfg)

    # Determine name of run and unique id (if not resuming)
    run_name = (
        f"{cfg.run_name}_h{cfg.model.hidden_size}_l{cfg.model.num_layers}_{cfg.seed}"
    )

    if job_type == "pretrain":
        annotations = False
        pretrain = True
    elif job_type == "nlu":
        assert best_model_path is not None, "Must provide best_model_path for nlu"
        run_name += "_nlu"
        annotations = True
        pretrain = False
    else:
        raise NotImplementedError(f"job_type {job_type} not implemented")

    resume_training = False
    unique_run_name = f"{cfg.seed}_{wandb.util.generate_id()}"
    checkpoint_path = f"{cfg.paths.output_dir}/checkpoints/{run_name}"
    if resume_from_checkpoint is not None:
        resume_training = True
        unique_run_name = resume_from_checkpoint

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
        data_dir_path=cfg.paths.input_dir,
        batch_size=cfg[job_type].batch_size,
        images=cfg.images,
        annotations=annotations,
        num_workers=cfg.num_workers,
        label_name_embeddings=cfg.model.label_name_embeddings,
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
            no_symbolic=cfg.model.no_symbolic,
            label_name_embeddings=cfg.model.label_name_embeddings,
        )

    print("Creating Trainer")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_path,
        every_n_epochs=1,
        filename=unique_run_name,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=10)

    trainer = pl.Trainer(
        max_epochs=cfg[job_type].max_epochs,
        logger=wandb_logger,
        gpus=cfg.gpus,
        callbacks=[early_stopping_callback, checkpoint_callback],
        fast_dev_run=cfg.fast,
        # strategy=DDPStrategy(find_unused_parameters=cfg.model.no_symbolic),
        strategy=DDPStrategy(find_unused_parameters=True),
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
