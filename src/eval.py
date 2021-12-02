import torch
import numpy as np
import torch.nn.functional as F
from time import time
from src.utils import create_dict_meters, AverageMeter

def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = torch.flip(np.argsort(sims[index]), [0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = torch.flip(np.argsort(sims[5 * index + i]), [0])
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def evaluate(image_embedder, text_embedder, loader, loss_fn, device, epoch=None):
    # We will do test on all five captions
    image_embedder.eval()
    text_embedder.eval()
    if epoch == None:
        metrics = create_dict_meters()
    else:
        loss_meter = AverageMeter()
    time_meter = AverageMeter()
    len_ = len(loader)
    start = time()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            # Extract positive captions
            ids = data["ids"].to(device, dtype=torch.long)
            masks = data["mask"].to(device, dtype=torch.long)
            if epoch is None:
                ids = torch.flatten(ids, 0, 1)
                masks = torch.flatten(masks, 0, 1)
            # Extract images
            input_images = data["image"].to(device, dtype=torch.float)

            # Compute embeddings for images and texts
            image_embeds = image_embedder(input_images)
            text_embeds = text_embedder(ids, masks)

            if epoch == None:

                image_embeds_norm = F.normalize(image_embeds, dim=1)
                text_embeds_norm = F.normalize(text_embeds, dim=1)
                sim_matrix = torch.mm(image_embeds_norm, text_embeds_norm.T)
                #i2t scores 
                (i2tr1, i2tr5, i2tr10, i2tmedr, i2tmeanr) = i2t(
                                                                image_embeds_norm, 
                                                                text_embeds_norm, 
                                                                None, 
                                                                sim_matrix, 
                                                                npts = None, 
                                                                return_ranks = False)
                metrics["i2t_meanr"].update(i2tmeanr)
                metrics["i2t_medr"].update(i2tmedr)
                metrics["i2t_r@10"].update(i2tr10)
                metrics["i2t_r@5"].update(i2tr5)
                metrics["i2t_r@1"].update(i2tr1)

                #i2t scores 
                (t2ir1, t2ir5, t2ir10, t2imedr, t2imeanr) = t2i(
                                                                image_embeds_norm, 
                                                                text_embeds_norm, 
                                                                None, 
                                                                sim_matrix, 
                                                                npts = None, 
                                                                return_ranks = False)

                metrics["t2i_meanr"].update(t2imeanr)
                metrics["t2i_medr"].update(t2imedr)
                metrics["t2i_r@10"].update(t2ir10)
                metrics["t2i_r@5"].update(t2ir5)
                metrics["t2i_r@1"].update(t2ir1)

            else:
                loss = loss_fn(image_embeds, text_embeds)
                loss_meter.update(loss.item())

            curr_iter = time() - start
            time_meter.update(time() - start)
            if epoch == None:
                print(
                    "TEST Batch [{}] | {:.2f}/{:.2f} [{:.2f} s/it]".format(
                        idx,
                        time_meter.sum,
                        time_meter.avg * len_,
                        time_meter.avg,
                    )
                )
            else:
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

    if epoch == None:
        final_metrics = " | ".join(
            ["{}: {:.3f}".format(name, metrics[name].avg) for name in metrics]
        )
        print("Final TEST metrics | " + final_metrics)
        return metrics
    else:
        return loss_meter.avg
