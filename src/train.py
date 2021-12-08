from time import time

import torch
from src.utils import AverageMeter

# device = "cuda" if cuda.is_available() else "cpu"
# max_len = data_qoutations.quotation.str.split().str.len().max()


def train_one_epoch(
    epoch,
    image_embedder,
    text_embedder,
    loss_fn,
    loader,
    optimizer,
    device,
    TRAINABLE_CV,
    TRAINABLE_TEXT,
):
    if TRAINABLE_CV == None:
        image_embedder.eval()
    else:
        image_embedder.train()
    if TRAINABLE_TEXT == None:
        text_embedder.eval()
    else:
        text_embedder.train()   
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    len_ = len(loader)
    start = time()
    for idx, data in enumerate(loader):
        optimizer.zero_grad()
        # Extract positive captions
        ids = data["ids"][0].to(device)
        masks = data["mask"][0].to(device)
        # Extract images
        input_images = data["image"][0].to(device)
        # Compute embeddings for images and texts
        image_embeds = image_embedder(input_images)
        text_embeds = text_embedder(ids, masks)
        # Loss computation
        loss = loss_fn(image_embeds, text_embeds)
        loss_meter.update(loss.item())
        # Backward
        loss.backward()
        optimizer.step()
        # Time measurement and print
        curr_iter = time() - start
        time_meter.update(curr_iter)
        print(
            "TRAIN Epoch {} [{}] | Loss {:.3f} ({:.3f}) | {:.2f}/{:.2f} [{:.2f} s/it]".format(
                epoch,
                idx,
                loss_meter.val,
                loss_meter.avg,
                time_meter.sum,
                time_meter.avg * len_,
                time_meter.avg,
            )
        )
        start += curr_iter
    return loss_meter.avg
