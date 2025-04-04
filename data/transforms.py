import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops


class CompleteGraph(BaseTransform):
    """
    This transform adds all pairwise edges into the edge index per data sample,
    then removes self loops, i.e. it builds a fully connected or complete graph
    """
    def __call__(self, data):
        
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data
    
    
class ConcatenateGlobal(BaseTransform):
    """_summary_

    Args:
        BaseTransform (_type_): _description_
    """
    
    def __call__(self, data):
        u = torch.concatenate([data.u_chem, data.u_dm], dim=0).view(1, -1)
        del data.u_dm
        del data.u_chem
        data.u = u
        
        return data