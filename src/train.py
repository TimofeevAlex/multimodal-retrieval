import torch
from torch import cuda
from tqdm import tqdm
from src.eval import compute_ranks, recall_k
from src.utils import AverageMeter

# device = "cuda" if cuda.is_available() else "cpu"
# max_len = data_qoutations.quotation.str.split().str.len().max()


def train_one_epoch(epoch, image_embedder, text_embedder, loss_fn, loader, optimizer, device):
    image_embedder.train()
    text_embedder.train()
    # ks = [1]
    # metrics = create_dict_meters(ks)
    loss_meter = AverageMeter()
    for data in loader:
        optimizer.zero_grad()
        # Extract positive captions
        ids = data["ids"].to(device, dtype=torch.long)
        masks = data["mask"].to(device, dtype=torch.long)
        # Extract images
        input_images = data["image"].to(device, dtype=torch.float)
        # Compute embeddings for images and texts
        image_embeds = image_embedder(input_images)
        text_embeds = text_embedder(ids, masks)
        # Loss computation
        loss = loss_fn(image_embeds, text_embeds)
        loss_meter.update(loss.item())
        # Backward
        loss.backward()
        optimizer.step()
        # # Metrics
        # image_ranks, text_ranks = compute_ranks(image_embeds, text_embeds, device)
        # metrics['mr_t2i'].update(torch.mean(torch.Tensor.float(text_ranks)))
        # metrics['mr_i2t'].update(torch.mean(torch.Tensor.float(image_ranks)))
        # for k in ks:
        #     image_recall, text_recall = recall_k(k, image_ranks, text_ranks)
        #     metrics[f'r@{k}_t2i'].update(image_recall)
        #     metrics[f'r@{k}_i2t'].update(text_recall)

    print('TRAIN Epoch {} | Loss {%.3f}'.format(epoch, loss_meter.avg))
    return loss_meter.avg
