from argparse import ArgumentParser

import pytorch_lightning as pl


from models.model import Piglet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from dataset import PigPenDataModule


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
    pigpen = PigPenDataModule(
        data_dir=hparams.input_dir, batch_size=hparams.batch_size, images=False
    )

    print("Creating Model")
    model = Piglet(
        reverse_object_mapping_dir=hparams.input_dir,
        hidden_size=hparams.hidden_size,
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        dropout=hparams.dropout,
    )

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
        "--max-epochs", default=2, type=int, help="max number of training epochs"
    )
    parser.add_argument(
        "--hidden-size", default=256, type=int, help="number of hidden units per layer"
    )
    parser.add_argument(
        "--num-layers", default=3, type=int, help="number of layers per sub model"
    )
    parser.add_argument(
        "--num-heads", default=4, type=int, help="number of heads in each transformer"
    )
    parser.add_argument(
        "--bert-model",
        default="roberta-base",
        type=str,
        help="LM model to use as action encoder",
    )
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout parameter")
    parser.add_argument("--fast", action="store_true", help="fast run for testing")

    args = parser.parse_args()

    seed_everything(int(args.seed))
    main(args)
