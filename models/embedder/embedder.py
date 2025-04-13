import torch

from torch_geometric.nn import MLP

import config as cfg


class FeatureEmbedder(torch.nn.Module):
    """
    FeatureEmbedder class.
    This class is used to embed the node features and the molecule features. using MLPs.
    The input features are the atom features and the molecule features.
    The output features are the embedded features.
    """

    def __init__(self, input_mol_emb_dim=cfg.IN_MOL_DIM, input_node_emb_dim=cfg.NODE_DIM, out_emb_dim=cfg.EMB_DIM):
        super(FeatureEmbedder, self).__init__()
        self.mlp_u_emb = MLP(
            in_channels=input_mol_emb_dim,
            hidden_channels=32,
            out_channels=out_emb_dim,
            num_layers=1,
            act=torch.nn.SiLU(),
            dropout=0.5,
            norm="LayerNorm",
        )

        self.mlp_x_emb = MLP(
            in_channels=input_node_emb_dim,
            hidden_channels=32,
            out_channels=out_emb_dim,
            num_layers=1,
            act=torch.nn.SiLU(),
            dropout=0.5,
            norm="LayerNorm",
        )

    def forward(self, batch):
        batch.u = self.mlp_u_emb(batch.u)
        batch.x = self.mlp_x_emb(batch.x)
        return batch
