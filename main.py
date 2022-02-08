from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import wandb

from dataset import PigPenDataset
from model import Piglet


def main(hparams):
    print(hparams.job_type)
    wandb_logger = WandbLogger(
        project="hog",
        entity="itl",
        job_type=hparams.job_type,
        config=hparams,
        save_dir=f"{hparams.output_dir}",
    )

    run_name = wandb_logger.experiment.name

    print("Loading dataset")
    train_dataset = PigPenDataset(data_dir=hparams.input_dir, data_split="train")
    val_dataset = PigPenDataset(data_dir=hparams.input_dir, data_split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    print("Creating Model")
    model = Piglet(reverse_object_mapping_dir=hparams.input_dir)

    print("Creating Trainer")
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", dirpath=f"{hparams.output_dir}/checkpoints/{run_name}"
    )
    if hparams.fast:
        trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            logger=wandb_logger,
            gpus=hparams.gpus,
            callbacks=[checkpoint_callback],
            limit_train_batches=10,
            limit_val_batches=10,
            log_every_n_steps=1,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=hparams.max_epochs,
            logger=wandb_logger,
            gpus=hparams.gpus,
            callbacks=[checkpoint_callback],
        )

    print("Training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_loader])


if __name__ == "__main__":
    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir", type=str, help="path to directory containing input data"
    )
    parser.add_argument(
        "--output-dir", type=str, help="path to directory to save output data"
    )
    parser.add_argument("--gpus", default=None, help="gpu ids to use")
    parser.add_argument(
        "--job-type", default="base", type=str, help="job type for wandb"
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument(
        "--max-epochs", default=20, type=int, help="max number of training epochs"
    )
    parser.add_argument("--fast", action="store_true", help="fast run for testing")

    args = parser.parse_args()

    main(args)
