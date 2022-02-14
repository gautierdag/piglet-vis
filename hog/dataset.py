import random
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Channel-wise pixel statistics collected over entire dataset
MEAN = torch.tensor([0.49458119, 0.43375753, 0.34601003])
STD = torch.tensor([0.19409628, 0.19771467, 0.19638838])


def denormalize_image(image):
    denormed = image * STD[:, None, None] + MEAN[:, None, None]
    denormed = denormed.permute(1, 2, 0)
    return denormed


def load_image_from_path(path: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path)) / 255.0)


class PigPenDataset(Dataset):
    def __init__(
        self,
        data_dir="../data",
        data_split: Optional[Literal["train", "val", "test"]] = "train",
        images=False,
        annotations=False,
        randomise_annotations=False,
    ):
        """
        Args:
            data_dir: directory containing the data
            data_split: train, val or test
            images: if True, return the images as well as the action/object vectors
            annotations: if True, return the annotations as well as the action/object vectors
            randomise_annotations: if True, randomly select one of the three annotations
        """

        self.data_split = data_split
        self.images = images
        self.annotations = annotations
        self.randomise_annotations = randomise_annotations

        self.action_matrix = np.load(f"{data_dir}/actions_{data_split}.npy")
        self.objects_matrix = np.load(f"{data_dir}/objects_{data_split}.npy")
        assert len(self.action_matrix) == len(self.objects_matrix)

        if self.images:
            image_indices_file = f"{data_dir}/img_indices_{data_split}.npy"
            self.image_directory = f"{data_dir}/{data_split}"

            self.image_indices = np.load(image_indices_file)
            assert len(self.action_matrix) == len(self.action_matrix)
            self.transforms = transforms.Compose([transforms.Normalize(MEAN, STD)])

        if self.annotations:
            self.precondition_text = np.load(
                f"{data_dir}/precondition_language_{data_split}.npy", allow_pickle=True
            )
            self.action_text = np.load(
                f"{data_dir}/action_language_{data_split}.npy", allow_pickle=True
            )
            self.postcondition_text = np.load(
                f"{data_dir}/postcondition_language_{data_split}.npy", allow_pickle=True
            )

            # 3 annotators per example
            assert self.precondition_text.shape[1] == 3
            assert self.action_text.shape[1] == 3
            assert self.postcondition_text.shape[1] == 3

    def __len__(self):
        return len(self.action_matrix)

    def __getitem__(self, index: int):
        action_vector = self.action_matrix[index]
        objects_vector = self.objects_matrix[index]

        item = {
            "actions": torch.from_numpy(action_vector),
            "objects": torch.from_numpy(objects_vector),
        }

        if self.images:
            image_0 = load_image_from_path(
                f"{self.image_directory}/{self.image_indices[index]}/0.jpeg"
            )
            image_1 = load_image_from_path(
                f"{self.image_directory}/{self.image_indices[index]}/1.jpeg"
            )

            # Permute for channel first then normalize
            image_0 = self.transforms(image_0.permute(-1, 0, 1))
            image_1 = self.transforms(image_1.permute(-1, 0, 1))
            # stack images
            item["images"] = torch.stack([image_0, image_1]).float()

        if self.annotations:
            # Loads raw text -> note that this is yet to be tokenized at this stage
            annotation_index = 0
            if self.randomise_annotations:
                annotation_index = random.randint(0, 2)

            item["precondition_text"] = self.precondition_text[index][annotation_index]
            item["action_text"] = self.action_text[index][annotation_index]
            item["postcondition_text"] = self.postcondition_text[index][
                annotation_index
            ]

        return item


def collate_fn_generator(
    tokenizer: Optional[Tokenizer],
) -> Callable[
    [List[Dict[str, Union[str, torch.Tensor]]]],
    Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
]:
    """
    Returns a collate function to be used with a dataloader
    A collate function is where individual examples get collated into a batch
    In most cases this is a simple stack, but in the case that we have a text input we must tokenize
    and pad appropriately.
    """

    def collate_fn(
        batch: List[Dict[str, Union[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        items = {}
        for key in batch[0].keys():
            if "text" in key and tokenizer is not None:
                items[key] = tokenizer(
                    [batch_item[key] for batch_item in batch],
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
            else:
                items[key] = torch.stack([batch_item[key] for batch_item in batch])
        return items

    return collate_fn


class PigPenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../data",
        batch_size: int = 32,
        images=False,
        annotations=False,
        randomise_annotations=False,
        bert_model: str = "roberta-base",
        num_workers=4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.images = images
        self.annotations = annotations
        self.randomise_annotations = randomise_annotations
        self.bert_model = bert_model
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """
        Make assignments here (val/train/test split)
        stage is a string for modes: fit/predict/validate/test
        """
        self.pigpen_train = PigPenDataset(
            data_dir=self.data_dir,
            data_split="train",
            images=self.images,
            annotations=self.annotations,
            randomise_annotations=self.randomise_annotations,
        )
        self.pigpen_val = PigPenDataset(
            data_dir=self.data_dir,
            data_split="val",
            images=self.images,
            annotations=self.annotations,
            randomise_annotations=self.randomise_annotations,
        )
        self.collate_fn = None
        if self.annotations:
            tokenizer = AutoTokenizer.from_pretrained(
                self.bert_model,
                cache_dir=f"output/bert-models/{self.bert_model}",
                model_max_length=512,
            )
            self.collate_fn = collate_fn_generator(tokenizer)
            self.pigpen_test = PigPenDataset(
                data_dir=self.data_dir,
                data_split="test",
                images=self.images,
                annotations=self.annotations,
                randomise_annotations=self.randomise_annotations,
            )

    def train_dataloader(self):
        return DataLoader(
            self.pigpen_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.pigpen_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        assert self.annotations, "Test set not available without annotations"
        return DataLoader(
            self.pigpen_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
        )
