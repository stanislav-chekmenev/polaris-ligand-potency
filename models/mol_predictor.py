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
        self.emb_gat = torch.nn.Sequential(
            torch.nn.Linear(
                cfg.EMB_DIM,
                cfg.EMB_DIM,
            ),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
        )
        self.emb_mace = torch.nn.Sequential(
            torch.nn.Linear(
                cfg.EMB_DIM,
                cfg.EMB_DIM,
            ),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
        )
        self.emb_bary = torch.nn.Sequential(
            torch.nn.Linear(
                cfg.EMB_DIM,
                cfg.EMB_DIM,
            ),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
        )
        self.lin_out = torch.nn.Sequential(
            torch.nn.Linear(cfg.EMB_DIM * 4, cfg.EMB_DIM),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(cfg.EMB_DIM, cfg.PREDICTION_DIM),
        )

    def forward(self, batch):
        # Embed the features
        batch = self.feature_embedder(batch)
        h_mol = batch.u
        del batch.u  # free up memory

        # Get GAT features -> dim = [batch_size, out_emb_dim]
        h_gat = self.gat(batch)
        # Embed the GAT features to bring them closer to the h_mol features in the embeddings space
        # Use a skip connection
        h_gat = h_gat + self.emb_gat(h_gat)

        # Get the MACE 3d features of dim = [batch_size, num_nodes, out_emb_dim]
        # and the barycenters of dim = [batch_size, out_emb_dim]
        batch, barycenters = self.mace_bary(batch)

        # Rebatch the batch tensor to be able to average across molecules and NOT conformers
        batch.batch = self.rebatch(batch) if not cfg.DEBUG_ONLY_ONE_CONFORMER else batch.batch
        h_mace = self.aggr(batch.h_mace, batch.batch)
        h_bary = self.aggr(barycenters.x, barycenters.batch)

        # Embed the MACE and barycenter features to bring them closer to the h_mol features in the embeddings space
        h_mace = h_mace + self.emb_mace(h_mace)
        h_bary = h_bary + self.emb_bary(h_bary)

        # Concatenate all embeddings and reshape to [batch_size, num_mol_emb, out_emb_dim]
        x = torch.cat([h_mol, h_gat, h_mace, h_bary], dim=-1).view(cfg.BATCH_SIZE, 4, cfg.EMB_DIM)

        # Apply the transformer block
        x = self.transformer(x).view(cfg.BATCH_SIZE, cfg.EMB_DIM * 4)

        # Apply the linear layer for predictions
        return {"pred": self.lin_out(x), "mol_emb": h_mol, "gat_emb": h_gat, "mace_emb": h_mace}

    def rebatch(self, batch):
        new_batch = batch.batch.clone()
        batch_range = range(cfg.BATCH_SIZE)

        # Get the indices for the molecules in the batch -> e.g. [0, 0, 0, 1, 1, 1] for batch size = 2 and 3 conformers
        mol_indices = np.repeat(np.array(list(batch_range)), cfg.NUM_CONFORMERS_SAMPLE).tolist()

        for conformer_idx, mol_idx in zip(range(batch.num_graphs), mol_indices):
            new_batch[new_batch == conformer_idx] = mol_idx

        return new_batch
