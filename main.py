import os
import os.path as osp
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.datasets as dset
from src import eval, loader, loss, model, train
from torch import cuda
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if cuda.is_available() else "cpu"


def run_train(
    DATA_DIRECTORY,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIRECTORY,
    LOSS,
    writer,
):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    annot_train = osp.join("annotations", "captions_train2014.json")
    # Define train loader
    tr_dataset = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "train2014"),
        annFile=osp.join(DATA_DIRECTORY, annot_train),
        transform=loader.get_transform("train"),
    )[:-5000]
    params = {"batch_size": BATCH_SIZE, "shuffle": False}
    train_dataset = loader.ImgCaptLoader(tr_dataset, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, **params)
    # Define val loader
    val_dataset = dset.CocoCaptions(
        root=osp.join(DATA_DIRECTORY, "train2014"),
        annFile=osp.join(DATA_DIRECTORY, annot_train),
        transform=loader.get_transform("val"),
    )[-5000:]
    val_dataset = loader.ImgCaptSetLoader(val_dataset, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, **params)

    # Initialize models
    text_embedder = model.DistilBERT(finetune=True, embedding_size=512).to(device)
    image_embedder = model.ResNet34(finetune=False).to(device)

    # Define loss function
    if LOSS == "triplet":
        loss_fn = loss.HingeTripletRankingLoss(
            margin=0.2, device=device, mining_negatives="max"
        ).to(device)
    elif LOSS == "SimCLR":
        loss_fn = loss.SimCLRLoss(temp=0.07, device=device).to(device)
    else:
        raise ValueError("Loss can be triplet/SimCLR")

    params = list(filter(lambda p: p.requires_grad, image_embedder.parameters()))
    params += list(filter(lambda p: p.requires_grad, text_embedder.parameters()))

    optimizer = torch.optim.Adam(params=params, lr=LEARNING_RATE)
    print('Start training')
    for epoch in range(EPOCHS):
        loss_tr = train.train_one_epoch(
            epoch, image_embedder, text_embedder, loss_fn, train_loader, optimizer, device
        )
        # Add loss to tensorboard
        writer.add_scalar('loss/train', loss_tr, epoch)
        if epoch % 5 == 0:
            val_metrics = eval.evaluate(
                image_embedder, text_embedder, loss_fn, val_loader, [1, 5, 10], device, epoch
            )
            # Add metrics to tensorboard
            for name in val_metrics:
                writer.add_scalar(name + '/val', val_metrics[name], epoch)

    torch.save(text_embedder.state_dict(), osp.join(OUTPUT_DIRECTORY, "text_embedder"))
    torch.save(image_embedder.state_dict(), osp.join(OUTPUT_DIRECTORY, "image_embedder")
    )
    return image_embedder, text_embedder


def run_test(
    DATA_DIRECTORY, MAX_LEN, BATCH_SIZE, LOSS, image_embedder, text_embedder, writer
):
    im_dir = osp.join(DATA_DIRECTORY, "val2014")
    annot_val = osp.join("annotations", "captions_val2014.json")
    cap_file = osp.join(DATA_DIRECTORY, annot_val)

    print("Running evaluations")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    test_dataset = dset.CocoCaptions(
        root=im_dir, annFile=cap_file, transform=loader.get_transform("test")
    )[:5000]
    test_dataset = loader.ImgCaptSetLoader(test_dataset, tokenizer, MAX_LEN)
    test_params = {"batch_size": BATCH_SIZE, "shuffle": False}
    # validation_dataset = torch.utils.data.Subset(validation_dataset, torch.arange(1024))
    test_loader = DataLoader(test_dataset, **test_params)

    if LOSS == "triplet":
            loss_fn = loss.HingeTripletRankingLoss(
            margin=0.2, device=device, mining_negatives="max"
        ).to(device)
    elif LOSS == "SimCLR":
        loss_fn = loss.SimCLRLoss(temp=0.07, device=device).to(device)
    else:
        raise ValueError("Loss can be triplet/SimCLR")
    
    eval.evaluate(image_embedder, text_embedder, loss_fn, test_loader, [1, 5, 10], device)


def read_embedders(input_directory):
    print("Loading models from input directory")
    text_embedder = model.DistilBERT(finetune=True, embedding_size=512).to(device)
    image_embedder = model.ResNet34(finetune=False).to(device)
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
        description="Takes the DATA_DIRECTORY, MAX_LEN, BATCH_SIZE, OUTPUT_DIRECTORY "
        "and trains and evaluates the model.",
    )

    parser.add_argument("--DOWNLOAD", type=str2bool, default=False)

    parser.add_argument("--DATA_DIRECTORY", type=str, default="dataset")

    parser.add_argument("--MAX_LEN", type=int, default=512)

    parser.add_argument("--EPOCHS", type=int, default=64)

    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    
    parser.add_argument("--BATCH_SIZE", type=int, default=128)

    parser.add_argument("--OUTPUT_DIRECTORY", type=str, default="model")

    parser.add_argument("--INPUT_DIRECTORY", type=str, default="")

    parser.add_argument("--LOSS", type=str, default="triplet")

    options = parser.parse_args()

    # Download MS-COCO
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
    now = str(datetime.now()).split('.')[0]
    exp_name = f"{options.LOSS}_{options.EPOCHS}_{options.LEARNING_RATE}_{options.BATCH_SIZE}_{now}"
    log_dir = 'logs'
    try:
        os.mkdir(log_dir)
    except:
        pass
    writer = SummaryWriter(osp.join(log_dir, exp_name))

    # Run training
    if options.INPUT_DIRECTORY == "":
        image_embedder, text_embedder = run_train(
            options.DATA_DIRECTORY,
            options.MAX_LEN,
            options.BATCH_SIZE,
            options.EPOCHS,
            options.LEARNING_RATE,
            options.OUTPUT_DIRECTORY,
            options.LOSS,
            writer
        )
    else:
        image_embedder, text_embedder = read_embedders(options.INPUT_DIRECTORY)

    # Test trained models
    run_test(
        options.DATA_DIRECTORY,
        options.MAX_LEN,
        options.BATCH_SIZE,
        options.LOSS,
        image_embedder,
        text_embedder,
        writer
    )
    writer.close()


if __name__ == "__main__":
    main()
