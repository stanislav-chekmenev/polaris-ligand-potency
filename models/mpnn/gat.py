import torch

from torch_geometric.nn import GATConv, global_add_pool

import config as cfg


class GAT(torch.nn.Module):
    def __init__(self, out_channels: int = 64, edge_dim: int = cfg.EDGE_DIM):
        """
        Args:
            out_channels (int): The number of output/hidden channels to update node features.
            edge_dim (int): The dimension of edge features.
        """
        super().__init__()

        self.gat_conv1 = GATConv(in_channels=-1, out_channels=out_channels, edge_dim=edge_dim)
        self.gat_conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, edge_dim=edge_dim)
        self.node_aggregation = global_add_pool

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): The input data object.
        Returns:
            torch.Tensor: The output tensor of node embeddings.
        """
        x, edge_index, edge_attr, batch = *data.x, data.edge_index, data.edge_attr, data.batch
        h = self.gat_conv1(x, edge_index, edge_attr)
        h = self.gat_conv2(h, edge_index, edge_attr)
        h = self.node_aggregation(h, batch)
        return h
