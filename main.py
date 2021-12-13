import argparse
import os
import os.path as osp
import sys
from datetime import datetime

import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
from src import eval, loader, loss, model, test, train
from torch import cuda
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import DistilBertTokenizerFast

# Fix seeds
seed = 7
torch.manual_seed(seed)
random.seed(0)
np.random.seed(0)

device = "cuda" if cuda.is_available() else "cpu"


def adjust_learning_rate(optimizer, epoch, init_lr):
    epoch = epoch + 1
    if epoch <= 5:
        lr = init_lr - (6 - epoch) * 0.00018 
    elif epoch > 90:
        lr = init_lr * 0.01
    elif epoch > 30:
        lr = init_lr * 0.1
    else:
        lr = init_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def run_train(
    DATA_DIRECTORY,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    OUTPUT_DIRECTORY,
    LOSS,
    TRAINABLE_CV,
    TRAINABLE_TEXT,
    writer,
    EMBEDDING_SIZE,
    SCHEDULER,
    OPTIMIZER
):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    annot_train = osp.join("annotations", "captions_train2014.json")
    # Define train loader
    tr_dataset = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "train2014"),
        annFile=osp.join(DATA_DIRECTORY, annot_train),
        transform=loader.get_transform("train"),
    )
    annot_val = osp.join("annotations", "captions_val2014.json")
    tr_dataset_ext = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "val2014"), 
        annFile=osp.join(DATA_DIRECTORY, annot_val), 
        transform=loader.get_transform("train")
    )
    params = {"batch_size": 1, "shuffle": True}
    len_ = len(tr_dataset)
    train_size = len_ - 5000
    indices = torch.arange(train_size)
    train_dataset = loader.ImgCaptLoader(
        tr_dataset,
        tokenizer,
        MAX_LEN,
        BATCH_SIZE,
        indices=indices,
        dataset_ext=tr_dataset_ext,
        sample_pos=True,
        shuffle=True,
    )
    # train_dataset = torch.utils.data.Subset(
    #     train_dataset, torch.arange(train_size))
    train_loader = DataLoader(train_dataset, **params)
    # Define val loader
    params = {"batch_size": 1, "shuffle": False}
    val_dataset = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "train2014"),
        annFile=osp.join(DATA_DIRECTORY, annot_train),
        transform=loader.get_transform("val"),
    )
    indices = torch.arange(train_size, len_)
    val_dataset = loader.ImgCaptLoader(
        val_dataset, tokenizer, MAX_LEN, BATCH_SIZE, indices=indices
    )
    # val_indices = torch.arange(train_size, len_)
    # val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_dataset, **params)

    # Initialize models
    text_embedder = model.DistilBERT(
        finetune=TRAINABLE_TEXT, embedding_size=EMBEDDING_SIZE
    ).to(device)
    image_embedder = model.ResNet(
        finetune=TRAINABLE_CV, embedding_size=EMBEDDING_SIZE
    ).to(device)

    # Define loss function
    if LOSS == "triplet":
        loss_fn = loss.HingeTripletRankingLoss(
            margin=0.2, device=device, mining_negatives="max"
        ).to(device)
    elif LOSS == "SimCLR":
        loss_fn = loss.SimCLRLoss(temp=0.07, device=device).to(device)
    elif LOSS == "BarlowTwins":
        loss_fn = loss.BarlowTwins(embedding_size=EMBEDDING_SIZE)
    else:
        raise ValueError("Loss can be triplet/SimCLR")

    params = list(filter(lambda p: p.requires_grad, image_embedder.parameters()))
    params += list(filter(lambda p: p.requires_grad, text_embedder.parameters()))

    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(
            params=params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(
            params=params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9
        )
    else:
        raise ValueError('Only Adam or SGD optimizers are acceptable')
        
    
    if SCHEDULER == 'MultiStep':
        scheduler = MultiStepLR(optimizer, milestones=[1, 4], gamma=0.1)
    elif SCHEDULER == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, len(train_loader))
    elif SCHEDULER == 'GradualWarmup':
        scheduler = None
    else:
        raise ValueError('Only MultiStep, CosineAnnealing, or GradualWarmup schedulers are acceptable')
        

    create_dir(OUTPUT_DIRECTORY)
    models_dir = osp.join(
        OUTPUT_DIRECTORY,
        f"{LOSS}_{LEARNING_RATE}_{WEIGHT_DECAY}_{EMBEDDING_SIZE}"
        +f"_{SCHEDULER}_{TRAINABLE_CV}_{TRAINABLE_TEXT}_{OPTIMIZER}"
        + str(datetime.now()).split(".")[0].replace(" ", "_"),
    )
    create_dir(models_dir)

    print("Start training")
    best_epoch = 0
    best_val_loss = int(sys.maxsize)
    for epoch in range(EPOCHS):
        if SCHEDULER == 'GradualWarmup':
            adjust_learning_rate(optimizer, epoch, LEARNING_RATE)
        # train one epoch
        loss_tr = train.train_one_epoch(
            epoch,
            image_embedder,
            text_embedder,
            loss_fn,
            train_loader,
            optimizer,
            device,
            TRAINABLE_CV,
            TRAINABLE_TEXT,
        )
        # Add loss to tensorboard
        # validate one epoch
        loss_val = eval.evaluate(
            image_embedder,
            text_embedder,
            val_loader,
            loss_fn,
            device,
            epoch,
        )
        # Add metrics to tensorboard
        writer.add_scalars("loss", {"train": loss_tr, "val": loss_val}, epoch)
        if SCHEDULER != 'GradualWarmup':
            scheduler.step()
        if loss_val <= best_val_loss:
            best_val_loss = loss_val
            best_epoch = epoch
        # Save the model
        torch.save(
            text_embedder.state_dict(),
            osp.join(models_dir, f"text_embedder_{epoch}"),
        )
        torch.save(
            image_embedder.state_dict(),
            osp.join(models_dir, f"image_embedder_{epoch}"),
        )
    # Upload the best model weights
    best_text_model_path = osp.join(models_dir, f"text_embedder_{best_epoch}")
    best_image_model_path = osp.join(models_dir, f"image_embedder_{best_epoch}")
    text_embedder.load_state_dict(torch.load(best_text_model_path))
    image_embedder.load_state_dict(torch.load(best_image_model_path))
    return image_embedder, text_embedder


def run_test(
    DATA_DIRECTORY,
    MAX_LEN,
    image_embedder,
    text_embedder,
    writer,
):
    im_dir = osp.join(DATA_DIRECTORY, "val2014")
    annot_val = osp.join("annotations", "captions_val2014.json")
    cap_file = osp.join(DATA_DIRECTORY, annot_val)

    print("Running evaluations")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    test_dataset = dset.CocoCaptions(
        root=im_dir, annFile=cap_file, transform=loader.get_transform("test")
    )
    test_dataset_i2t = loader.ImgCaptSetLoader(
        test_dataset, tokenizer, MAX_LEN, i2t=True
    )
    test_dataset_t2i = loader.ImgCaptSetLoader(
        test_dataset, tokenizer, MAX_LEN, i2t=False
    )
    test_params = {"batch_size": 1, "shuffle": False}
    test_loader_i2t = DataLoader(test_dataset_i2t, **test_params)
    test_loader_t2i = DataLoader(test_dataset_t2i, **test_params)

    metrics_t2i, metrics_i2t = test.test(
        image_embedder, text_embedder, test_loader_t2i, test_loader_i2t, device
    )
    writer.add_scalars("metrics/t2i", metrics_t2i)
    writer.add_scalars("metrics/i2t", metrics_i2t)


def read_embedders(path_cv, path_text, TRAINABLE_CV, TRAINABLE_TEXT, EMBEDDING_SIZE):
    print("Loading models from input directory")
    text_embedder = model.DistilBERT(
        finetune=TRAINABLE_TEXT, embedding_size=EMBEDDING_SIZE
    ).to(device)
    text_embedder.load_state_dict(torch.load(path_text))
    image_embedder = model.ResNet(
        finetune=TRAINABLE_CV, embedding_size=EMBEDDING_SIZE
    ).to(device)
    image_embedder.load_state_dict(torch.load(path_cv))
    return image_embedder, text_embedder


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def create_dir(path):
    try:
        os.mkdir(path)
    except:
        pass


def main() -> None:
    """Runs the export"""

    parser = argparse.ArgumentParser(
        "main",
        description="Takes the DATA_DIRECTORY, MAX_LEN, BATCH_SIZE, OUTPUT_DIRECTORY "
        "and trains and evaluates the model.",
    )

    parser.add_argument("--DOWNLOAD", type=str2bool, default=False)
    parser.add_argument("--DATA_DIRECTORY", type=str, default="dataset")
    parser.add_argument("--MAX_LEN", type=int, default=32)
    parser.add_argument("--EPOCHS", type=int, default=60)
    parser.add_argument("--LEARNING_RATE", type=float, default=2e-4)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=5e-6)
    parser.add_argument("--BATCH_SIZE", type=int, default=256)
    parser.add_argument("--OUTPUT_DIRECTORY", type=str, default="model")
    parser.add_argument("--CV_DIR", type=str, default="")
    parser.add_argument("--TEXT_DIR", type=str, default="")
    parser.add_argument("--LOSS", type=str, default="triplet")
    parser.add_argument("--TRAINABLE_CV", type=str, default="all")
    parser.add_argument("--TRAINABLE_TEXT", type=str, default="all")
    parser.add_argument("--EMBEDDING_SIZE", type=int, default=128)
    parser.add_argument("--SCHEDULER", type=str, default='MultiStep')
    parser.add_argument("--OPTIMIZER", type=str, default='Adam')
    parser.add_argument("--RESTART", type=str2bool, default=False)


    options = parser.parse_args()

    # Download MS-COCO
    if options.DOWNLOAD:
        from zipfile import ZipFile

        import wget

        # create a dir for the dataset
        create_dir(options.DATA_DIRECTORY)
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
            if "annotations" in file:
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

    # Create a writer for tensorboard
    now = str(datetime.now()).split(".")[0].replace(" ", "_")
    log_dir = "logs"
    create_dir(log_dir)
    # Run training or load models for testing
    if (options.CV_DIR == "") or (options.TEXT_DIR == ""):
        exp_name = (
            f"TRAIN_{options.LOSS}_{options.EPOCHS}_{options.LEARNING_RATE}_{options.SCHEDULER}"
            + f"_{options.WEIGHT_DECAY}_{options.BATCH_SIZE}_{options.EMBEDDING_SIZE}_{options.OPTIMIZER}_{now}"
        )
        writer = SummaryWriter(osp.join(log_dir, exp_name))
        if options.RESTART:
            assert options.CV_DIR != ""
            assert options.TEXT_DIR != ""
            image_embedder, text_embedder = read_embedders(
                options.CV_DIR,
                options.TEXT_DIR,
                options.TRAINABLE_CV,
                options.TRAINABLE_TEXT,
                options.EMBEDDING_SIZE
            )
        image_embedder, text_embedder = run_train(
            options.DATA_DIRECTORY,
            options.MAX_LEN,
            options.BATCH_SIZE,
            options.EPOCHS,
            options.LEARNING_RATE,
            options.WEIGHT_DECAY,
            options.OUTPUT_DIRECTORY,
            options.LOSS,
            options.TRAINABLE_CV,
            options.TRAINABLE_TEXT,
            writer,
            options.EMBEDDING_SIZE,
            options.SCHEDULER,
            options.OPTIMIZER
        )
    else:
        exp_name = (
            f"TEST_{options.LOSS}_{options.EPOCHS}_{options.LEARNING_RATE}_{options.SCHEDULER}"
            + f"_{options.WEIGHT_DECAY}_{options.BATCH_SIZE}_{options.EMBEDDING_SIZE}_{options.OPTIMIZER}_{now}"
        )
        writer = SummaryWriter(osp.join(log_dir, exp_name))
        image_embedder, text_embedder = read_embedders(
            options.CV_DIR,
            options.TEXT_DIR,
            options.TRAINABLE_CV,
            options.TRAINABLE_TEXT,
            options.EMBEDDING_SIZE,
        )

    # Test trained models
    run_test(
        options.DATA_DIRECTORY,
        options.MAX_LEN,
        image_embedder,
        text_embedder,
        writer,
    )
    writer.close()


if __name__ == "__main__":
    main()
