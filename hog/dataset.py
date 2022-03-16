import random
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DetrFeatureExtractor

PigPenExample = Dict[str, Union[str, torch.Tensor]]

# Channel-wise pixel statistics collected over entire dataset
# MEAN = torch.tensor([0.49458119, 0.43375753, 0.34601003])
# STD = torch.tensor([0.19409628, 0.19771467, 0.19638838])

MEAN = torch.tensor([0.485, 0.456, 0.406])  # image net MEAN
STD = torch.tensor([0.229, 0.224, 0.225])  # image net STD


def denormalize_image(image):
    denormed = image * STD[:, None, None] + MEAN[:, None, None]
    return denormed


def load_image_from_path(path: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path)) / 255.0).float()


class PigPenDataset(Dataset):
    def __init__(
        self,
        data_dir_path="../data",
        data_split: Optional[Literal["train", "val", "test"]] = "train",
        images=False,
        annotations=False,
        randomise_annotations=False,
        indices=False,
    ):
        """
        Args:
            data_dir_path: directory containing the data
            data_split: train, val or test
            images: if True, return the images as well as the action/object vectors
            annotations: if True, return the annotations as well as the action/object vectors
            randomise_annotations: if True, randomly select one of the three annotations
        """

        self.data_split = data_split
        self.images = images
        self.annotations = annotations
        self.randomise_annotations = randomise_annotations
        self.indices = indices

        self.action_matrix = np.load(f"{data_dir_path}/actions_{data_split}.npy")
        self.objects_matrix = np.load(f"{data_dir_path}/objects_{data_split}.npy")
        assert len(self.action_matrix) == len(self.objects_matrix)

        if self.images:
            image_indices_file = f"{data_dir_path}/img_indices_{data_split}.npy"
            self.image_directory = f"{data_dir_path}/{data_split}"

            self.image_indices = np.load(image_indices_file)
            assert len(self.action_matrix) == len(self.image_indices)

        if self.annotations:
            self.precondition_text = np.load(
                f"{data_dir_path}/precondition_language_{data_split}.npy",
                allow_pickle=True,
            )
            self.action_text = np.load(
                f"{data_dir_path}/action_language_{data_split}.npy", allow_pickle=True
            )
            self.postcondition_text = np.load(
                f"{data_dir_path}/postcondition_language_{data_split}.npy",
                allow_pickle=True,
            )

            # 3 annotators per example
            assert self.precondition_text.shape[1] == 3
            assert self.action_text.shape[1] == 3
            assert self.postcondition_text.shape[1] == 3

    def __len__(self):
        return len(self.action_matrix)

    def __getitem__(self, index: int) -> PigPenExample:
        action_vector = self.action_matrix[index]
        objects_vector = self.objects_matrix[index]

        item: PigPenExample = {
            "actions": torch.from_numpy(action_vector),
            "objects": torch.from_numpy(objects_vector),
        }

        if self.images:
            image_0 = load_image_from_path(
                f"{self.image_directory}/{self.image_indices[index]}_0.jpeg"
            )
            image_1 = load_image_from_path(
                f"{self.image_directory}/{self.image_indices[index]}_1.jpeg"
            )
            # stack images
            images = torch.stack([image_0, image_1])
            images = rearrange(images, "i h w c -> i c h w", c=3)
            item["images"] = images

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
        if self.indices:
            item["indices"] = torch.tensor(self.image_indices[index])

        return item


def collate_fn_generator(
    tokenizer: Optional[Tokenizer] = None,
    image_feature_extractor: Optional[
        Callable[[List[torch.Tensor]], Dict[str, torch.Tensor]]
    ] = None,
) -> Callable[
    [List[PigPenExample]],
    Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
]:
    """
    Args:
        tokenizer: tokenizer to use for text
        image_feature_extractor: image feature extractor to use for images
                                 note this is what huggingface uses to denote
                                 the transformation and preparation step before the image
                                 is passed to the vision model

    Returns a collate function to be used with a dataloader
    A collate function is where individual examples get collated into a batch
    In most cases this is a simple stack, but in the case that we have a text input we must tokenize
    and pad appropriately.
    """

    def collate_fn(
        batch: List[PigPenExample],
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        items: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        for key in batch[0].keys():
            if "text" in key:
                assert tokenizer is not None, "tokenizer not provided"
                items[key] = tokenizer(
                    [batch_item[key] for batch_item in batch],
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
            elif "images" in key:
                assert (
                    image_feature_extractor is not None
                ), "image feature extractor not provided"
                images = torch.stack([batch_item[key] for batch_item in batch])
                images = rearrange(images, "b i c h w -> (b i) c h w", c=3, i=2)
                items[key] = image_feature_extractor(list(images), return_tensors="pt")[
                    "pixel_values"
                ]
            else:
                items[key] = torch.stack([batch_item[key] for batch_item in batch])

        return items

    return collate_fn


class PigPenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir_path: str = "../data",
        output_dir_path: str = "../output",
        batch_size: int = 32,
        images=False,
        annotations=False,
        randomise_annotations=False,
        bert_model: str = "roberta-base",
        vision_model: str = "detr",
        indices=False,
        num_workers=4,
    ):
        super().__init__()
        self.data_dir_path = data_dir_path
        self.output_dir_path = output_dir_path
        self.batch_size = batch_size
        self.images = images
        self.annotations = annotations
        self.randomise_annotations = randomise_annotations
        self.bert_model = bert_model
        self.vision_model = vision_model
        self.num_workers = num_workers
        self.indices = indices

    def setup(self, stage: Optional[str] = None):
        """
        Make assignments here (val/train/test split)
        stage is a string for modes: fit/predict/validate/test
        """
        self.pigpen_train = PigPenDataset(
            data_dir_path=self.data_dir_path,
            data_split="train",
            images=self.images,
            annotations=self.annotations,
            randomise_annotations=self.randomise_annotations,
            indices=self.indices,
        )
        self.pigpen_val = PigPenDataset(
            data_dir_path=self.data_dir_path,
            data_split="val",
            images=self.images,
            annotations=self.annotations,
            randomise_annotations=self.randomise_annotations,
            indices=self.indices,
        )
        self.collate_fn = None
        tokenizer = None
        image_feature_extractor = None
        # if using annotations then we will need to tokenize the text
        # and it also means that we have access to a test split
        if self.annotations:
            tokenizer = AutoTokenizer.from_pretrained(
                self.bert_model,
                cache_dir=f"{self.output_dir_path}/bert-models/{self.bert_model}",
                model_max_length=512,
            )
            self.pigpen_test = PigPenDataset(
                data_dir_path=self.data_dir_path,
                data_split="test",
                images=self.images,
                annotations=self.annotations,
                randomise_annotations=self.randomise_annotations,
            )

        # if using vision then we need the transform and load operation to apply to images
        if self.images:
            if self.vision_model == "detr":
                image_feature_extractor = DetrFeatureExtractor.from_pretrained(
                    "facebook/detr-resnet-50",
                    cache_dir=f"{self.output_dir_path}/vision_model/detr",
                    do_resize=False,
                    do_normalize=True,
                )
            else:
                raise NotImplemented(f"Image model {self.vision_model} not implemented")

        if tokenizer is not None or image_feature_extractor is not None:
            self.collate_fn = collate_fn_generator(
                tokenizer=tokenizer, image_feature_extractor=image_feature_extractor
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
