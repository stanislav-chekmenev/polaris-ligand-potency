import numpy as np
import torch

import config as cfg

from models import FeatureEmbedder, GAT, MACEBaryModel, TransformerBlock


class MolPredictor(torch.nn.Module):
    """
    MolPredictor class.
    This class is used to implement a molecular property predictor using MACE, GAT, FGW barycenter aggregation
    and a Transformer for models' embeddings.
    """

    def __init__(self, aggr=cfg.NODE_AGGREGATION):
        """
        Initialize the MolPredictor class.
        """
        super().__init__()
        self.feature_embedder = FeatureEmbedder()
        self.mace_bary = MACEBaryModel(mace_kwargs=cfg.MACE_KWARGS)
        self.transformer = TransformerBlock()
        self.gat = GAT()
        self.aggr = aggr
        self.lin_out = torch.nn.Linear(cfg.EMB_DIM, cfg.PREDICTION_DIM)

    def forward(self, batch):
        # Embed the features 
        batch = self.feature_embedder(batch)
        h_mol = batch.u
        del batch.u # free up memory

        # Get GAT features -> dim = [batch_size, out_emb_dim]
        h_gat = self.gat(batch)

        # Get the MACE 3d features of dim = [batch_size, num_nodes, out_emb_dim]
        # and the barycenters of dim = [batch_size, out_emb_dim]
        if batch.pos is not None:
            batch, barycenters = self.mace_bary(batch)
        else:
            pass

        # Aggregate MACE 3D features accross all nodes in the batch. 
        # [batch_size, num_nodes, out_emb_dim] -> [batch_size, out_emb_dim]
        new_batch = self.rebatch(batch)
        h_mace = self.aggr(batch.h_mace, new_batch) 
        
        # Concatenate all embeddings
        x = torch.cat([h_mol, h_gat, h_mace, barycenters.x], dim=-1)

        # Apply the transformer block
        x = self.transformer(x)

        # Apply the linear layer for predictions
        return self.lin_out(x)

    def rebatch(self, batch):
        new_batch = batch.batch.clone()
        batch_range = range(cfg.BATCH_SIZE)

        # Get the indices for the molecules in the batch -> [0, 0, 0, 1, 1, 1] for batch size = 2
        mol_idx = np.repeat(np.array(list(batch_range)), cfg.NUM_CONFORMERS_SAMPLE).tolist()

        for conformer_idx, mol_idx in zip(range(batch.num_graphs), mol_idx):
            new_batch[new_batch == conformer_idx] = mol_idx

        return new_batch





















'''
    def forward(self, batch):
        """
        Forward pass of the MolPredictor class.
        Args:
            batch (torch_geometric.data.Batch): The input batch of data.
        Returns:
            torch.Tensor: The output tensor of the model.
        """
        # Embed the features
        batch = self.feature_embedder(batch)
        h_mol = batch.u
        del batch.u

        # Get GAT features -> dim = [batch_size, out_emb_dim]
        h_gat = self.gat(batch)
        
        # Get the MACE 3d features of dim = [batch_size, num_nodes, out_emb_dim]
        # and the barycenters of dim = [batch_size, out_emb_dim]
        batch, barycenters = self.mace_bary(batch)

        # Aggregate MACE 3D features accross all nodes in the batch
        new_batch = batch.batch
        mol_idx = np.repeat(np.array(cfg.BATCH_SIZE), cfg.NUM_CONFORMERS_SAMPLE).tolist()
        
        for conformer_idx, mol_idx in zip(range(batch.num_graphs), mol_idx):
            new_batch[new_batch == conformer_idx] = mol_idx

        batch.batch = new_batch
        batch.ptr = torch.zeros(cfg.BATCH_SIZE + 1, dtype=torch.long)

        h_mace = self.aggr(batch.h_mace, batch.batch)

        # Concatenate all embeddings
        x = torch.cat([h_mol, h_gat, h_mace, barycenters.x], dim=-1)
        
        # Apply the transformer block
        x = self.transformer(x)
        
        
        return x
'''