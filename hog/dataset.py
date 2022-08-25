import os
import random
import re
from functools import lru_cache
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import HogConfig
from models.image_models import BoundingBoxImageModel, get_image_feature_extractor
from models.mappings import get_actions_mapper, get_objects_mapper

PigPenExample = Dict[str, Union[str, torch.Tensor]]

# Channel-wise pixel statistics collected over entire dataset (unused)
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
        images_raw=False,
        annotations=False,
        label_name_embeddings=False,
        randomise_annotations=False,
        use_full=False,
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
        self.h5py_file_path = f"{data_dir_path}/piglet.h5"
        self.h5py_dataset_path = f"{data_split}"
        self.annotations = annotations
        self.label_name_embeddings = label_name_embeddings
        full = ""
        if use_full:
            full = "_full"

        if self.annotations:
            data_dir_path += "/annotated"

        self.image_directory = f"{data_dir_path}/{data_split}"
        self.images_raw = images_raw
        image_indices_file = f"{data_dir_path}/img_indices_{data_split}{full}.npy"
        self.image_indices = np.load(image_indices_file)

        self.randomise_annotations = randomise_annotations

        self.action_matrix = np.load(f"{data_dir_path}/actions_{data_split}{full}.npy")

        # seen matrix controls what objects to show during training/validation
        self.seen_matrix = np.load(f"{data_dir_path}/seen_{data_split}{full}.npy")
        if self.data_split != "test":
            # if not test then we only select indices where True
            self.seen_matrix = np.where(self.seen_matrix)[0]

        self.objects_matrix = np.load(f"{data_dir_path}/objects_{data_split}{full}.npy")
        assert len(self.action_matrix) == len(self.objects_matrix)
        if self.images_raw:
            assert len(self.action_matrix) == len(self.image_indices)

        if self.images or self.label_name_embeddings:
            assert os.path.exists(self.h5py_file_path), "h5py file does not exist"
            self.h5 = h5py.File(self.h5py_file_path, "r", libver="latest", swmr=True)

        if self.annotations:
            self.h5py_dataset_path += "/annotated"
            self.precondition_text = np.load(
                f"{data_dir_path}/precondition_language_{data_split}{full}.npy",
                allow_pickle=True,
            )
            self.action_text = np.load(
                f"{data_dir_path}/action_language_{data_split}{full}.npy",
                allow_pickle=True,
            )
            self.postcondition_text = np.load(
                f"{data_dir_path}/postcondition_language_{data_split}{full}.npy",
                allow_pickle=True,
            )

            # 3 annotators per example
            assert self.precondition_text.shape[1] == 3
            assert self.action_text.shape[1] == 3
            assert self.postcondition_text.shape[1] == 3

    def __len__(self):
        return len(self.seen_matrix)

    def __getitem__(self, index: int) -> PigPenExample:
        h5_index = index
        if self.data_split != "test":
            index = self.seen_matrix[index]

        action_vector = self.action_matrix[index]
        objects_vector = self.objects_matrix[index]

        item: PigPenExample = {
            "actions": torch.from_numpy(action_vector),
            "objects": torch.from_numpy(objects_vector),
        }

        if self.images:
            hid_pre = self.h5[f"{self.h5py_dataset_path}/hidden/pre"][h5_index]
            hid_post = self.h5[f"{self.h5py_dataset_path}/hidden/post"][h5_index]
            hid_pre = torch.tensor(hid_pre)
            hid_post = torch.tensor(hid_post)
            item["images_hidden_states"] = torch.stack([hid_pre, hid_post])

        if self.images_raw:
            image_0 = load_image_from_path(
                f"{self.image_directory}/{self.image_indices[index]}_0.jpeg"
            )
            image_1 = load_image_from_path(
                f"{self.image_directory}/{self.image_indices[index]}_1.jpeg"
            )
            # stack images
            images = torch.stack([image_0, image_1])
            images = rearrange(images, "i h w c -> i c h w", c=3)
            item["images_raw"] = images

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

        if self.label_name_embeddings:
            obj1_name = torch.from_numpy(
                self.h5["object_name_label_embeddings"][objects_vector[0, 0]]
            )
            obj2_name = torch.from_numpy(
                self.h5["object_name_label_embeddings"][objects_vector[1, 0]]
            )
            item["object_name_embeddings"] = torch.stack(
                [obj1_name, obj2_name, obj1_name, obj2_name]
            )

            action_name = torch.from_numpy(
                self.h5["action_name_label_embeddings"][action_vector[0]]
            )
            action_object_name = torch.from_numpy(
                self.h5["object_name_label_embeddings"][action_vector[1]]
            )
            action_receptacle_name = torch.from_numpy(
                self.h5["object_name_label_embeddings"][action_vector[2]]
            )
            item["action_object_name_embeddings"] = torch.stack(
                [action_name, action_object_name, action_receptacle_name]
            )

        item["indices"] = torch.tensor(index)

        # track seen examples
        if self.data_split == "test":
            item["seen"] = torch.tensor(self.seen_matrix[index])

        return item

    @lru_cache(maxsize=32)
    def get_images_and_bounding_boxes(
        self, index: int
    ) -> Tuple[
        TensorType[2, "height", "width", "channel"], TensorType[2, "num_boxes", 4]
    ]:
        image_0 = load_image_from_path(
            f"{self.image_directory}/{self.image_indices[index]}_0.jpeg"
        )
        image_1 = load_image_from_path(
            f"{self.image_directory}/{self.image_indices[index]}_1.jpeg"
        )
        images = torch.stack([image_0, image_1])

        with h5py.File(self.h5py_file_path, "r", libver="latest", swmr=True) as h5:
            bboxes_pre = self.h5[f"{self.h5py_dataset_path}/bboxes/pre"][index]
            bboxes_post = self.h5[f"{self.h5py_dataset_path}/bboxes/post"][index]
            bboxes_pre = torch.tensor(bboxes_pre)
            bboxes_post = torch.tensor(bboxes_post)
            bboxes = torch.stack([bboxes_pre, bboxes_post])

        return images, bboxes


def preprocess_label_embeddings(cfg: HogConfig, h5_file_path: str):
    print("Preprocessing object and action name embeddings")

    object_name_mapper = get_objects_mapper(cfg.paths.input_dir)[0]
    # split labels with Capital Letters into multiple words
    object_name_mapper = {
        k: re.sub(r"(\w)([A-Z])", r"\1 \2", v) for k, v in object_name_mapper.items()
    }

    actions_name_mapper = get_actions_mapper(cfg.paths.input_dir)
    actions_name_mapper = {
        k: re.sub(r"(\w)([A-Z])", r"\1 \2", v).replace("Object", "").strip()
        for k, v in actions_name_mapper.items()
    }

    #  load model and tokenizer
    bert_model_name = cfg.model.bert_model
    bert_model = AutoModel.from_pretrained(
        bert_model_name,
        cache_dir=f"{cfg.paths.output_dir}/bert-models/{bert_model_name}",
        add_pooling_layer=False,
    )
    bert_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        bert_model_name,
        cache_dir=f"{cfg.paths.output_dir}/bert-models/{bert_model_name}",
        model_max_length=512,
    )
    h5_file = h5py.File(h5_file_path, "a", libver="latest")

    ###### Object labels ######
    object_label_names = list(object_name_mapper.values())
    object_label_names_tokenized = tokenizer(
        object_label_names, padding=True, return_tensors="pt", truncation=True
    )
    # run BERT on all labels (done on cpu)
    with torch.no_grad():
        output = bert_model(**object_label_names_tokenized)
        output = output.last_hidden_state[:, 0, :]

    # save embeddings to h5 file
    object_name_label_embeddings = h5_file.create_dataset(
        "object_name_label_embeddings",
        shape=(len(object_label_names), bert_model.config.hidden_size),
        dtype=np.float32,
        fillvalue=0,
    )
    object_name_label_embeddings[:] = output.numpy()

    ###### Action labels ######
    action_label_names = list(actions_name_mapper.values())
    action_label_names_tokenized = tokenizer(
        action_label_names, padding=True, return_tensors="pt", truncation=True
    )
    # run BERT on all action labels (done on cpu)
    with torch.no_grad():
        output = bert_model(**action_label_names_tokenized)
        output = output.last_hidden_state[:, 0, :]

    # save embeddings to h5 file
    action_name_label_embeddings = h5_file.create_dataset(
        "action_name_label_embeddings",
        shape=(len(action_label_names), bert_model.config.hidden_size),
        dtype=np.float32,
        fillvalue=0,
    )
    action_name_label_embeddings[:] = output.numpy()

    h5_file.close()
    del bert_model


def preprocess_images(cfg: HogConfig):
    """
    Preprocess images for the model by running all datasets through the VisionModel
    Saves the output representations and bounding box coordinates to a h5 file
    This can take around 2-3 hours to run on a GPU
    If h5 file already exists then it is loaded and the preprocessing is skipped
    """
    h5_file_path = f"{cfg.paths.input_dir}/piglet.h5"
    if cfg.use_full:
        h5_file_path = f"{cfg.paths.input_dir}/piglet_full.h5"

    if os.path.exists(h5_file_path):
        h5_file = h5py.File(h5_file_path, "r", libver="latest", swmr=True)
        if "action_name_label_embeddings" not in h5_file:
            h5_file.close()
            preprocess_label_embeddings(cfg, h5_file_path)
        else:
            h5_file.close()
        return
    print("Preprocessing images")

    image_model = BoundingBoxImageModel(
        image_model_name="detr",
        output_dir_path=cfg.paths.output_dir,
    )
    device = "cpu"
    if torch.cuda.is_available():
        print("cuda available")
        device = "cuda"

    image_model = image_model.to(device)

    image_feature_extractor = get_image_feature_extractor(cfg.paths.output_dir)

    def image_only_collate(batch):
        images = torch.stack([batch_item["images_raw"] for batch_item in batch])
        images = rearrange(images, "b i c h w -> (b i) c h w", c=3, i=2)
        images = image_feature_extractor(list(images), return_tensors="pt")[
            "pixel_values"
        ]
        return images

    print(f"Creating h5 file at {h5_file_path}")
    h5_file = h5py.File(h5_file_path, "w", libver="latest")
    datasets = {
        f"{cfg.paths.input_dir}": ["train", "val"],
        f"{cfg.paths.input_dir}/annotated": ["train", "val", "test"],
    }
    for path, splits in datasets.items():
        for split in splits:
            base_path = f"{split}"
            if "annotated" in path:
                base_path += "/annotated"

            print(f"Extracting images for {base_path}")
            dataset = PigPenDataset(
                data_dir_path=path,
                images_raw=True,
                data_split=split,
                use_full=cfg.use_full,
            )
            batch_size = 32
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=image_only_collate,
            )

            dataset_length = len(dataset)
            hid_pre = h5_file.create_dataset(
                f"{base_path}/hidden/pre",
                shape=(dataset_length, 100, 256),
                dtype=np.float32,
                fillvalue=0,
            )
            hid_post = h5_file.create_dataset(
                f"{base_path}/hidden/post",
                shape=(dataset_length, 100, 256),
                dtype=np.float32,
                fillvalue=0,
            )
            bbox_pre = h5_file.create_dataset(
                f"{base_path}/bboxes/pre",
                shape=(dataset_length, 100, 4),
                dtype=np.int32,
                fillvalue=0,
            )
            bbox_post = h5_file.create_dataset(
                f"{base_path}/bboxes/post",
                shape=(dataset_length, 100, 4),
                dtype=np.int32,
                fillvalue=0,
            )
            for i, batch in tqdm(enumerate(loader), total=dataset_length // batch_size):
                with torch.no_grad():
                    batch = batch.to(device)
                    bboxes, hidden_state = image_model(batch)
                    bboxes = rearrange(
                        bboxes,
                        "(b i) n p -> b i n p",
                        n=100,
                        p=4,
                        i=2,
                    )
                    bboxes = bboxes.cpu().numpy()

                    hidden_state = rearrange(
                        hidden_state,
                        "(b i) n h -> b i n h",
                        n=100,
                        h=256,
                        i=2,
                    )
                    hidden_state = hidden_state.cpu().numpy()

                    hid_pre[i * batch_size : (i + 1) * batch_size] = hidden_state[
                        :, 0, :, :
                    ]
                    hid_post[i * batch_size : (i + 1) * batch_size] = hidden_state[
                        :, 1, :, :
                    ]
                    bbox_pre[i * batch_size : (i + 1) * batch_size] = bboxes[:, 0, :, :]
                    bbox_post[i * batch_size : (i + 1) * batch_size] = bboxes[
                        :, 1, :, :
                    ]
    h5_file.close()
    del image_model

    # run preprocessing of object_names
    preprocess_label_embeddings(cfg, h5_file_path)


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
            elif "images_raw" in key:
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
        label_name_embeddings=False,
        bert_model: str = "roberta-base",
        vision_model: str = "detr",
        num_workers=4,
        use_full: bool = False,
    ):
        super().__init__()
        self.data_dir_path = data_dir_path
        self.output_dir_path = output_dir_path
        self.batch_size = batch_size
        self.images = images
        self.annotations = annotations
        self.randomise_annotations = randomise_annotations
        self.label_name_embeddings = label_name_embeddings
        self.bert_model = bert_model
        self.vision_model = vision_model
        self.num_workers = num_workers
        self.use_full = use_full

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
            label_name_embeddings=self.label_name_embeddings,
            use_full=self.use_full,
        )
        self.pigpen_val = PigPenDataset(
            data_dir_path=self.data_dir_path,
            data_split="val",
            images=self.images,
            annotations=self.annotations,
            randomise_annotations=self.randomise_annotations,
            label_name_embeddings=self.label_name_embeddings,
            use_full=self.use_full,
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
                label_name_embeddings=self.label_name_embeddings,
                use_full=self.use_full,
            )

        # if using vision then we need the transform and load operation to apply to images
        if self.images:
            image_feature_extractor = get_image_feature_extractor(self.output_dir_path)
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
