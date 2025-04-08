import torch

from torch_geometric.nn import GATConv, global_add_pool

import config as cfg


class GAT(torch.nn.Module):
    def __init__(self, in_channels=cfg.IN_GAT_DIM, out_channels: int = cfg.IN_GAT_DIM, edge_dim: int = cfg.EDGE_DIM):
        super().__init__()

        self.gat_conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim)
        self.gat_conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, edge_dim=edge_dim)
        self.act = torch.nn.SiLU()
        self.aggr = cfg.GAT_NODE_AGGREGATION

    def forward(self, batch):
        """
        Args:
            data (torch_geometric.data.Data): The input data object.
        Returns:
            torch.Tensor: The output tensor of node embeddings.
        """
        h = self.gat_conv1(batch.x, batch.edge_index, batch.edge_attr)
        h = self.act(h)
        h = self.gat_conv2(h, batch.edge_index, batch.edge_attr)
        h = self.act(h)
        h = self.aggr(h, batch.batch)
        return h
