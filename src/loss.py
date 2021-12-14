import torch
import torch.nn.functional as F
from torch import cuda, nn

# device = "cuda" if cuda.is_available() else "cpu"


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
        mask = torch.eye(batch_size).bool().to(self.device)

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
        mask = torch.eye(batch_size).bool().to(self.device)

        image_embeds_norm = F.normalize(image_embeds, dim=1)
        text_embeds_norm = F.normalize(text_embeds, dim=1)

        sim_matrix = torch.mm(image_embeds_norm, text_embeds_norm.T)
        sim_matrix = torch.exp(torch.div(sim_matrix, self.temp))

        pos_sim = sim_matrix.diag()
        neg_sim = sim_matrix * (
            torch.ones(batch_size, batch_size).to(self.device)
            - torch.eye(batch_size, batch_size).to(self.device)
        )

        image_term = torch.log(pos_sim / neg_sim.sum(0))
        text_term = torch.log(pos_sim / neg_sim.T.sum(0))

        loss = -torch.mean(image_term + text_term)
        return loss


class BarlowTwins(nn.Module):
    def __init__(self, embedding_size, lambd=5e-3):
        super().__init__()
        # normalization layer for the representations image_embeds and text_embeds
        self.bn = nn.BatchNorm1d(embedding_size, affine=False)
        self.lambd = lambd
    
    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, image_embeds, text_embeds):
        batch_size = image_embeds.shape[0]
        # empirical cross-correlation matrix
        sim_matrix = self.bn(image_embeds).T @ self.bn(text_embeds)
        # sum the cross-correlation matrix between all gpus
        sim_matrix.div_(batch_size)
        on_diag = torch.diagonal(sim_matrix).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(sim_matrix).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
