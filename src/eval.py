from time import time

import torch
from src.utils import AverageMeter


def evaluate(image_embedder, text_embedder, loader, loss_fn, device, epoch):
    # We will do test on all five captions
    image_embedder.eval()
    text_embedder.eval()
    loss_meter = AverageMeter()
    time_meter = AverageMeter()
    len_ = len(loader)
    start = time()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            # Extract positive captions
            ids = data["ids"].to(device)
            masks = data["mask"].to(device)
            # Extract images
            input_images = data["image"].to(device)
            # Compute embeddings for images and texts
            image_embeds = image_embedder(input_images)
            text_embeds = text_embedder(ids, masks)
            # Compute loss
            loss = loss_fn(image_embeds, text_embeds)
            loss_meter.update(loss.item())
            # Log
            curr_iter = time() - start
            time_meter.update(time() - start)
            print(
                "VAL Epoch {} [{}] | Loss {:.3f} ({:.3f}) | {:.2f}/{:.2f} [{:.2f} s/it]".format(
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
