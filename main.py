from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from dataset import PigPenDataset
from model import Piglet


def main(hparams):
    print(hparams.job_type)
    wandb_logger = WandbLogger(
        project="hog",
        entity="itl",
        job_type=hparams.job_type,
        config=hparams,
    )

    print("Loading dataset")
    train_dataset = PigPenDataset(data_dir="data", data_split="train")
    val_dataset = PigPenDataset(data_dir="data", data_split="val")

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
    model = Piglet()

    print("Creating Trainer")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss")
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
    parser.add_argument("--gpus", default=None, help="gpu ids to use")
    parser.add_argument(
        "--job-type", default="base", type=str, help="job type for wandb"
    )
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument(
        "--max-epochs", default=20, type=int, help="max number of training epochs"
    )
    args = parser.parse_args()

    main(args)
