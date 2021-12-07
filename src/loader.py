import math
import numpy as np
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
        t_list = [transforms.RandomResizedCrop(
            224), transforms.RandomHorizontalFlip()]
    elif split_name == "val":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == "test":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform


class ImgCaptSetLoader(Dataset):
    def __init__(self, dataset, tokenizer, max_len, batch_size=250, num_images=5000, i2t=True):
        assert num_images % batch_size == 0
        self.tokenizer = tokenizer
        self.coco_dataset = dataset
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_images = num_images
        self.num_captions = num_images * 5
        self.i2t = i2t
        # Prepare dataset
        self.images = []
        self.captions = []
        for i, (img, cap) in enumerate(dataset):
            if i == num_images:
                break
            self.images.append(img)
            self.captions.extend(cap[:5])

    def __len__(self):
        return math.ceil(self.num_captions * self.num_images / self.batch_size ** 2) 

    def __getitem__(self, index):
        # Prepare indices
        if self.i2t:
            sample_index = self.batch_size * index
            caption_index = sample_index % self.num_captions
            image_index = (sample_index // self.num_captions) * self.batch_size
        else:
            sample_index = self.batch_size * index
            image_index = sample_index % self.num_images
            caption_index = (sample_index // self.num_images) * self.batch_size
        # Get this batch of images
        images = self.images[image_index:image_index + self.batch_size]
        captions = self.captions[caption_index:caption_index + self.batch_size]
        # Processing captions
        inputs = self.tokenizer.batch_encode_plus(
            captions,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image": torch.tensor(torch.stack(images), dtype=torch.float)
        }


class ImgCaptLoader(Dataset):
    def __init__(self, dataset, tokenizer, max_len, batch_size, indices=None, sample_pos=False, shuffle=False):
        self.tokenizer = tokenizer
        self.coco_dataset = dataset
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        if indices == None:
            self.indices = np.arange(len(dataset))
        else:
            self.indices= indices
        self.sample_pos = sample_pos
        
        self.images = []
        self.captions = []
        for i in indices:
            img, cap = dataset[i]
            self.images.append(img)
            self.captions.append(cap[:5])
        self.images = np.array(self.images) 
        self.captions = np.array(self.captions) 
        
        self.num_samples = len(self.images)
        self.epoch_indices = np.arange(self.num_samples)    
           
    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, index): 
        if self.shuffle and (index == 0):
            self.epoch_indices = np.random.permutation(self.epoch_indices)
        sample_index = index * self.batch_size
        cur_indices = self.epoch_indices[sample_index:sample_index + self.batch_size]
        images = self.images[cur_indices]
        captions = self.captions[cur_indices]
        if self.sample_pos:
            pos_samples = np.random.randint(0, 5, self.batch_size)
            captions = self.captions[:, pos_samples]
        else:
            captions = self.captions[:, 0]
        # Processing captions
        inputs = self.tokenizer.batch_encode_plus(
            captions,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image": torch.tensor(torch.stack(images), dtype=torch.float)
        }



# class ImgCaptLoader(Dataset):
#     def __init__(self, dataset, tokenizer, max_len):
#         self.tokenizer = tokenizer
#         self.coco_dataset = dataset
#         self.max_len = max_len
#         # If we are training only destillbert with initially frozen embeddings we can just preprocess the image embeddings and use them as target

#     def __len__(self):
#         return len(self.coco_dataset)

#     def __getitem__(self, index):
#         # We are loading only a single caption to enable easier batch negative sampling
#         img, captions = self.coco_dataset[
#             index
#         ]  # loading pair image - one caption (should they be shuffled here)?
#         inputs = self.tokenizer.encode_plus(
#             captions[random.randint(0, 4)],  # fetching the single caption
#             None,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding="max_length",
#             truncation=True,
#         )
#         ids = inputs["input_ids"]
#         mask = inputs["attention_mask"]

#         return {
#             "ids": torch.tensor(ids, dtype=torch.long),
#             "mask": torch.tensor(mask, dtype=torch.long),
#             "image": torch.tensor(img, dtype=torch.float)
#         }
