import random

import torch
import torchvision.transforms as transforms  # tested with transformers 4.12.5
from torch import cuda, nn
from torch.utils.data import Dataset

device = "cuda" if cuda.is_available() else "cpu"


def get_transform(split_name):
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    t_list = []
    if split_name == "train":
        t_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    elif split_name == "val":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == "test":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


class ImgCaptSetLoader(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.coco_dataset = dataset
        self.max_len = max_len
        # If we are training only destillbert with initially frozen embeddings we can just preprocess the image embeddings and use them as target

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        # We are loading only a single caption to enable easier batch negative sampling
        img, captions = self.coco_dataset[
            index
        ]  # loading pair image - one caption (should they be shuffled here)?
        inputs = [
            self.tokenizer.encode_plus(
                c,  # fetching the single caption
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
            )
            for c in captions[:5]
        ]
        ids = [input["input_ids"] for input in inputs]
        mask = [input["attention_mask"] for input in inputs]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image": img,  # float
        }


class ImgCaptLoader(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.coco_dataset = dataset
        self.max_len = max_len
        # If we are training only destillbert with initially frozen embeddings we can just preprocess the image embeddings and use them as target

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        # We are loading only a single caption to enable easier batch negative sampling
        img, captions = self.coco_dataset[
            index
        ]  # loading pair image - one caption (should they be shuffled here)?
        inputs = self.tokenizer.encode_plus(
            captions[0],  # fetching the single caption
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image": img,  # float
        }
