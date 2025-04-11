import torch

import config as cfg


class BaselineMLP(torch.nn.Module):

    def __init__(self, in_dim=cfg.IN_MOL_DIM, out_dim=cfg.PREDICTION_DIM):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, out_dim),
        )

    def forward(self, batch):
        x = batch.u
        x = self.mlp(x)
        return x
