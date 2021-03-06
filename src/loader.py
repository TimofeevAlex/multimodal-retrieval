import math
import numpy as np
import torch
import torchvision.transforms as transforms  # tested with transformers 4.12.5
from torch import cuda, nn
from torch.utils.data import Dataset
from tqdm import tqdm

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
    def __init__(
        self, dataset, tokenizer, max_len, batch_size=500, num_images=5000, i2t=True
    ):
        assert num_images % batch_size == 0
        assert batch_size % 5 == 0
        self.tokenizer = tokenizer
        self.coco_dataset = dataset
        self.max_len = max_len
        self.c_batch_size = batch_size
        self.i_batch_size = batch_size // 5
        self.num_images = num_images
        self.num_captions = num_images * 5
        self.i2t = i2t
        # Prepare dataset
        self.images = []
        self.captions = []
        print("Dataset loading")
        for i, (img, cap) in tqdm(enumerate(dataset)):
            if i == num_images:
                break
            self.images.append(img)
            self.captions.extend(cap[:5])

    def __len__(self):
        return math.ceil(
            (self.num_captions / self.c_batch_size)
            * (self.num_images / self.i_batch_size)
        )

    def __getitem__(self, index):
        # Prepare indices
        if self.i2t:
            caption_index = self.c_batch_size * index
            image_index = (caption_index // self.num_captions) * self.i_batch_size
            caption_index = caption_index % self.num_captions
        else:
            image_index = self.i_batch_size * index
            caption_index = (image_index // self.num_images) * self.c_batch_size
            image_index = image_index % self.num_images
        # Get this batch of images
        images = self.images[image_index : image_index + self.i_batch_size]
        captions = self.captions[caption_index : caption_index + self.c_batch_size]
        # Processing captions
        inputs = self.tokenizer.batch_encode_plus(
            captions,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image": torch.tensor(torch.stack(images), dtype=torch.float),
        }


class ImgCaptLoader(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_len,
        batch_size,
        indices=None,
        dataset_ext=None,
        sample_pos=False,
        shuffle=False,
    ):
        self.tokenizer = tokenizer
        # self.coco_dataset = np.array(dataset, dtype=object)[indices]
        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        if indices == None:
            self.indices = np.arange(len(dataset))
        else:
            self.indices = indices
        self.num_samples = len(self.indices)
        self.sample_pos = sample_pos

        self.images = []
        self.captions = []
        print("Dataset loading")
        for i in tqdm(self.indices):
            img, cap = dataset[i]
            self.images.append(np.array(img))
            self.captions.append(cap[:5])
        if dataset_ext != None:
            ext_len = len(dataset_ext)
            self.num_samples += ext_len - 5000
            for i in tqdm(np.arange(5000, ext_len)):
                img, cap = dataset_ext[i]
                self.images.append(np.array(img))
                self.captions.append(cap[:5])
        self.images = np.array(self.images)
        self.captions = np.array(self.captions)
        self.epoch_indices = np.arange(self.num_samples)

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, index):
        if self.shuffle and (index == 0):
            self.epoch_indices = np.random.permutation(self.epoch_indices)
        sample_index = index * self.batch_size
        cur_indices = self.epoch_indices[sample_index : sample_index + self.batch_size]
        images = self.images[cur_indices]
        if self.sample_pos:
            pos_samples = np.random.randint(0, 5, len(cur_indices))
            captions = self.captions[cur_indices, pos_samples]
        else:
            captions = self.captions[cur_indices, 0]
        # Processing captions
        inputs = self.tokenizer.batch_encode_plus(
            list(captions),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "image": torch.tensor(images, dtype=torch.float),
        }


if __name__ == "__main__":
    import os.path as osp
    import torchvision.datasets as dset
    from transformers import DistilBertTokenizerFast
    from torch.utils.data import DataLoader

    DATA_DIRECTORY = "dataset"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    annot_train = osp.join("annotations", "captions_train2014.json")
    # Define train loader
    tr_dataset = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "train2014"),
        annFile=osp.join(DATA_DIRECTORY, annot_train),
        transform=get_transform("train"),
    )
    params = {"batch_size": 1, "shuffle": True}
    train_size = 1000
    indices = torch.arange(train_size)
    train_dataset = ImgCaptLoader(
        tr_dataset,
        tokenizer,
        32,
        64,
        indices=indices,
        sample_pos=True,
        shuffle=True,
    )
    # train_dataset = torch.utils.data.Subset(
    #     train_dataset, torch.arange(train_size))
    train_loader = DataLoader(train_dataset, **params)
    for epoch in range(5):
        for batch in train_loader:
            pass
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
