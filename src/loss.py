import torch
import torch.nn.functional as F
from torch import cuda, nn

device = "cuda" if cuda.is_available() else "cpu"


class HingeTripletRankingLoss(nn.Module):
    def __init__(self, margin, device, mining_negatives="max"):
        super(HingeTripletRankingLoss, self).__init__()
        self.margin = margin
        self.device = device
        self.mining_negatives = mining_negatives

    def compute_one_term(self, sim_matrix, mask, pos_sim):
        neg_sim = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        neg_sim = F.relu(neg_sim.T - pos_sim + self.margin)
        if self.mining_negatives == "sum":
            neg_sim.sum(0)
        elif self.mining_negatives == "max":
            neg_sim.max(0)
        else:
            raise ValueError("mining_negatives must be either sum or max")
        return neg_sim

    def forward(self, image_embeds, text_embeds):
        batch_size = image_embeds.shape[0]
        mask = torch.eye(batch_size).bool().to(device)

        image_embeds_norm = F.normalize(image_embeds, dim=1)
        text_embeds_norm = F.normalize(text_embeds, dim=1)

        sim_matrix = torch.mm(image_embeds_norm, text_embeds_norm.T)
        pos_sim = sim_matrix.diag()

        image_term = self.compute_one_term(sim_matrix, mask, pos_sim)
        text_term = self.compute_one_term(sim_matrix.T, mask, pos_sim)

        loss = torch.mean(text_term + image_term)
        return loss


class SimCLRLoss(nn.Module):
    def __init__(self, temp, device):
        super(SimCLRLoss, self).__init__()
        self.temp = temp
        self.device = device

    def forward(self, image_embeds, text_embeds):
        batch_size = image_embeds.shape[0]
        mask = torch.eye(batch_size).bool().to(device)

        image_embeds_norm = F.normalize(image_embeds, dim=1)
        text_embeds_norm = F.normalize(text_embeds, dim=1)

        sim_matrix = torch.mm(image_embeds_norm, text_embeds_norm.T)
        sim_matrix = torch.exp(torch.div(sim_matrix, self.temp))

        pos_sim = sim_matrix.diag()
        neg_sim = sim_matrix * (
            torch.ones(batch_size, batch_size).to(device)
            - torch.eye(batch_size, batch_size).to(device)
        )

        image_term = torch.log(pos_sim / neg_sim.sum(0))
        text_term = torch.log(pos_sim / neg_sim.T.sum(0))

        loss = torch.mean(-image_term - text_term)
        return loss
