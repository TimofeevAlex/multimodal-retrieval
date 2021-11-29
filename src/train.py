import torch
from tqdm import tqdm
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
# max_len = data_qoutations.quotation.str.split().str.len().max()

def train_one_epoch(epoch, image_embedder, text_embedder, loss_fn, loader, optimizer):
    image_embedder.train()
    text_embedder.train()
    running_loss = 0.0
    for data in tqdm(loader):
        optimizer.zero_grad()
        # Extract positive captions
        ids = data['ids'].to(device, dtype = torch.long)
        masks = data['mask'].to(device, dtype = torch.long)
        # Extract images
        input_images = data['image'].to(device, dtype = torch.float)
        # Compute embeddings for images and texts
        image_embeds = image_embedder(input_images)
        text_embeds = text_embedder(ids, masks)
        # Loss computation
        loss = loss_fn(image_embeds, text_embeds)
        running_loss += loss.item()
        # Backward
        loss.backward()
        optimizer.step()

    return running_loss / len(loader)
