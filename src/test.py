from time import time

import numpy as np
import torch
import torch.nn.functional as F
from src.utils import AverageMeter


def compute_ranks_i2t(sims, start_index, npts):
    ranks = np.zeros((npts, 5))
    # top1 = np.zeros(npts)

    for index in range(npts):
        inds = torch.flip(np.argsort(sims[index]), [0])

        # Score
        rank = 1e20            
        for i in range(start_index + 5 * index, start_index + 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    return ranks.astype(int)


def compute_ranks_t2i(sims, start_index, npts):
    ranks = np.zeros(npts)
    # top1 = np.zeros(5 * npts)

    for index in range(npts):
        inds = torch.flip(np.argsort(sims[index]), [0])
        ranks[index] = np.where(inds == start_index + (index // 5))[0][0]
         # top1[5 * index + i] = inds[0]

    return ranks.astype(int)


def compute_metrics(ranks):
    # if num_relevants == 1:
    #     ranks = np.expand_dims(ranks, -1)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1, r5, r10, medr, meanr


def metrics_i2t(ranks):
    r1, r5, r10, medr, meanr = compute_metrics(ranks)
    return {
        "i2t_meanr": meanr,
        "i2t_medr": medr,
        "i2t_r@10": r10,
        "i2t_r@5": r5,
        "i2t_r@1": r1
    }


def metrics_t2i(ranks):
    r1, r5, r10, medr, meanr = compute_metrics(ranks)
    return {
        "t2i_meanr": meanr,
        "t2i_medr": medr,
        "t2i_r@10": r10,
        "t2i_r@5": r5,
        "t2i_r@1": r1
    }


def test_i2t(image_embedder, text_embedder, loader, device):
    image_embedder.eval()
    text_embedder.eval()

    time_meter = AverageMeter()
    len_ = len(loader)
    batch_size = loader.dataset.batch_size
    num_captions = loader.dataset.num_captions
    iters_per_image_batch = (num_captions // batch_size) - 1
    start = time()
    image_ranks = np.array([]).astype(int)
    with torch.no_grad():
        for idx, data in enumerate(loader):
            if idx % iters_per_image_batch == 0:
                if idx != 0:
                    # Store ranks for current
                    start_index = ((idx * batch_size) // num_captions) * batch_size
                    batch_image_ranks = compute_ranks_i2t(
                        sim_matrix, start_index, batch_size
                    )
                    image_ranks = np.append(image_ranks, batch_image_ranks)
                sim_matrix = torch.empty((batch_size, num_captions)).cpu()
                # Extract images
                input_images = data["image"][0].to(device)
                image_embeds = image_embedder(input_images)
                image_embeds_norm = F.normalize(image_embeds, dim=1)
            # Extract captions
            ids = data["ids"].to(device)
            masks = data["mask"].to(device)
            # Compute embeddings for images and texts
            text_embeds = text_embedder(ids[0], masks[0])
            text_embeds_norm = F.normalize(text_embeds, dim=1)
            # Batch similarities
            sample_index = (idx * batch_size) % num_captions
            sim_matrix[:, sample_index:sample_index + batch_size] = torch.mm(
                image_embeds_norm, text_embeds_norm.T
            ).cpu()
            # Logging
            curr_iter = time() - start
            time_meter.update(time() - start)
            print(
                "TEST I2T Batch [{}] | {:.2f}/{:.2f} [{:.2f} s/it]".format(
                    idx,
                    time_meter.sum,
                    time_meter.avg * len_,
                    time_meter.avg,
                )
            )
            start += curr_iter
    metrics = metrics_i2t(image_ranks)  
    return metrics


def test_t2i(image_embedder, text_embedder, loader, device):
    image_embedder.eval()
    text_embedder.eval()

    time_meter = AverageMeter()
    len_ = len(loader)
    batch_size = loader.dataset.batch_size
    num_images = loader.dataset.num_images
    iters_per_image_batch = (num_images // batch_size) - 1
    start = time()
    text_ranks = np.array([]).astype(int)
    with torch.no_grad():
        for idx, data in enumerate(loader):
            if idx  % iters_per_image_batch == 0:
                if idx != 0:
                    # Store ranks for current
                    start_index = ((idx * batch_size) // num_images) * batch_size / 5
                    batch_text_ranks = compute_ranks_t2i(
                        sim_matrix, start_index, batch_size
                    )
                    text_ranks = np.append(text_ranks, batch_text_ranks)
                sim_matrix = torch.empty((batch_size, num_images)).cpu()
                # Extract captions
                ids = data["ids"].to(device)
                masks = data["mask"].to(device)
                # Compute embeddings for images and texts
                text_embeds = text_embedder(ids[0], masks[0])
                text_embeds_norm = F.normalize(text_embeds, dim=1)
            # Extract images
            input_images = data["image"][0].to(device)
            image_embeds = image_embedder(input_images)
            image_embeds_norm = F.normalize(image_embeds, dim=1)
            # Batch similarities
            sample_index = (idx * batch_size) % num_images
            sim_matrix[:, sample_index:sample_index + batch_size] = torch.mm(
                text_embeds_norm, image_embeds_norm.T
            ).cpu()
            # Logging
            curr_iter = time() - start
            time_meter.update(time() - start)
            print(
                "TEST T2I Batch [{}] | {:.2f}/{:.2f} [{:.2f} s/it]".format(
                    idx,
                    time_meter.sum,
                    time_meter.avg * len_,
                    time_meter.avg,
                )
            )
            start += curr_iter
    metrics = metrics_t2i(text_ranks)
    return metrics


def test(image_embedder, text_embedder, loader_t2i, loader_i2t, device):
    metrics_val_t2i = test_t2i(image_embedder, text_embedder, loader_t2i, device)
    metrics_val_i2t = test_i2t(image_embedder, text_embedder, loader_i2t, device)
    final_metrics_t2i = " | ".join(
        ["{}: {:.3f}".format(name, metrics_val_t2i[name]) for name in metrics_val_t2i]
    )
    final_metrics_i2t = " | ".join(
        ["{}: {:.3f}".format(name, metrics_val_i2t[name]) for name in metrics_val_i2t]
    )
    print("Final TEST metrics")
    print('T2I | ' + final_metrics_t2i)
    print('I2T | ' + final_metrics_i2t)
    return metrics_val_t2i, metrics_val_i2t
