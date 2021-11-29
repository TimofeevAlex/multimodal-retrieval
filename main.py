import os
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torchvision.datasets as dset
from src import eval, loader, loss, model, train
from torch import cuda
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

device = "cuda" if cuda.is_available() else "cpu"
# import matplotlib.pyplot as plt


def run_train(
    DATA_DIRECTORY,
    MAX_LEN,
    TRAIN_BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIRECTORY,
    INPUT_DIRECTORY="",
    LOSS="triplet",
):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    annot_train = osp.join('annotations', "captions_train2014.json")
    tr_dataset = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "train2014"),
        annFile=osp.join(DATA_DIRECTORY, annot_train),
        transform=loader.get_transform("train"),
    )

    train_dataset = loader.ImgCaptLoader(tr_dataset, tokenizer, MAX_LEN)
    train_params = {"batch_size": TRAIN_BATCH_SIZE, "shuffle": False}
    # train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(1024))
    train_loader = DataLoader(train_dataset, **train_params)
    text_embedder = model.DistilBERT(768, finetune=False).to(device)
    image_embedder = model.ResNet34(768, finetune=False).to(device)

    if LOSS == "triplet":
        loss_fn = loss.HingeTripletRankingLoss(
            margin=0.2, device=device, mining_negatives="sum"
        ).to(device)
    elif LOSS == "SimCLR":
        loss_fn = loss.SimCLRLoss(temp=0.07, device=device).to(device)
    else:
        print("Loss can be triplet/SimCLR")
        return

    params = list(filter(lambda p: p.requires_grad, image_embedder.parameters()))
    params += list(filter(lambda p: p.requires_grad, text_embedder.parameters()))

    optimizer = torch.optim.Adam(params=params, lr=LEARNING_RATE)
    loss_vals = []
    for epoch in range(EPOCHS):
        loss_val = train.train_one_epoch(
            epoch, image_embedder, text_embedder, loss_fn, train_loader, optimizer
        )
        loss_vals.append(loss_val)

    # plt.plot(loss_vals)
    torch.save(text_embedder.state_dict(), osp.join(OUTPUT_DIRECTORY, "text_embedder"))
    torch.save(image_embedder.state_dict(), osp.join(OUTPUT_DIRECTORY, "image_embedder"))
    return image_embedder, text_embedder


def run_eval(
    DATA_DIRECTORY, MAX_LEN, VAL_BATCH_SIZE, image_embedder, text_embedder, split="val"
):
    if split == "val":
        im_dir = osp.join(DATA_DIRECTORY, "val2014")
        annot_val = osp.join('annotations', "captions_val2014.json")
        cap_file = osp.join(DATA_DIRECTORY, annot_val)
    else:
        raise ValueError('no annotations for test')
        # im_dir = DATA_DIRECTORY + "test2014/"
        # cap_file = (
        #     DATA_DIRECTORY + "captions_test2014.json"
        # )  # What are the test captions?
    print("Running evaluations")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    val_dataset = dset.CocoCaptions(
        root=im_dir, annFile=cap_file, transform=loader.get_transform("val")
    )
    validation_dataset = loader.ImgCaptSetLoader(val_dataset, tokenizer, MAX_LEN)
    val_params = {"batch_size": VAL_BATCH_SIZE, "shuffle": False}
    # validation_dataset = torch.utils.data.Subset(validation_dataset, torch.arange(1024))
    validation_loader = DataLoader(validation_dataset, **val_params)

    eval.evaluate(image_embedder, text_embedder, validation_loader, [1, 5, 10], device)


def read_embedders(input_directory):
    print("Loading models from input directory")
    text_embedder = model.DistilBERT(768, finetune=False).to(device)
    image_embedder = model.ResNet34(768, finetune=False).to(device)
    text_embedder.load_state_dict(torch.load(input_directory + "text_embedder"))
    image_embedder.load_state_dict(torch.load(input_directory + "image_embedder"))
    return image_embedder, text_embedder


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def main() -> None:
    """Runs the export"""

    parser = argparse.ArgumentParser(
        "main",
        description="Takes the DATA_DIRECTORY, MAX_LEN, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, OUTPUT_DIRECTORY "
        "and trains and evaluates the model.",
    )

    parser.add_argument("--DOWNLOAD", type=str2bool, default=False)

    parser.add_argument("--DATA_DIRECTORY", type=str, default="dataset")

    parser.add_argument("--MAX_LEN", type=int, default=512)

    parser.add_argument("--EPOCHS", type=int, default=64)

    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    parser.add_argument("--TRAIN_BATCH_SIZE", type=int, default=128)

    parser.add_argument("--VAL_BATCH_SIZE", type=int, default=128)

    parser.add_argument("--OUTPUT_DIRECTORY", type=str, default="model")

    parser.add_argument("--INPUT_DIRECTORY", type=str, default="")

    parser.add_argument("--LOSS", type=str, default="triplet")

    options = parser.parse_args()

    if options.DOWNLOAD:
        import wget
        from zipfile import ZipFile
        # create a dir for the dataset
        try:
            os.mkdir(options.DATA_DIRECTORY)
        except:
            pass
        # download files
        files = [
            "train2014.zip",
            "val2014.zip",
            "annotations_trainval2014.zip",
        ]
        for file in files:
            to_save = osp.join(options.DATA_DIRECTORY, file)
            if os.path.exists(to_save):
                continue
            if 'annotations' in file:
                url = f"http://images.cocodataset.org/annotations/{file}"
            else:
                url = f"http://images.cocodataset.org/zips/{file}"
            print(f"\nDownloading {to_save} from {url}")
            wget.download(url, to_save)
        # unzip files
        for file in files:
            to_save = osp.join(options.DATA_DIRECTORY, file)
            print(f"\nUnzipping {to_save}")
            with ZipFile(to_save, "r") as zip_obj:
                zip_obj.extractall(options.DATA_DIRECTORY)

    if options.INPUT_DIRECTORY == "":
        image_embedder, text_embedder = run_train(
            options.DATA_DIRECTORY,
            options.MAX_LEN,
            options.TRAIN_BATCH_SIZE,
            options.EPOCHS,
            options.LEARNING_RATE,
            options.OUTPUT_DIRECTORY,
            options.INPUT_DIRECTORY,
            options.LOSS,
        )
    else:
        image_embedder, text_embedder = read_embedders(options.INPUT_DIRECTORY)

    run_eval(
        options.DATA_DIRECTORY,
        options.MAX_LEN,
        options.VAL_BATCH_SIZE,
        image_embedder,
        text_embedder,
        split="val",
    )


if __name__ == "__main__":
    main()
