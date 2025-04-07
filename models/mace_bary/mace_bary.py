import logging
import numpy as np
import os
import torch

from ot.gromov import fgw_barycenters
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from typing import Tuple

import config as cfg

from models import MACEModel
from data.moldataset import MolDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MACEBaryModel(MACEModel):

    def __init__(self, mace_kwargs):
        super().__init__(**mace_kwargs)

    def forward(self, batch) -> Tuple[Batch, ...]:
        """
        Forward pass of the MACE model with barycenter computation.
        This function computes the forward pass of the MACE model and
        additionally computes the barycenter of the node features.
        
        Parameters:
            - batch: Batch of input data that has several molecules.
            - max_iter: Maximum number of iterations for barycenter computation.
            - epsilon: Convergence threshold for barycenter computation.
        """

        # Sample N number of conformers
        batch = self.sample_conformers(batch) 

        # Replicate the graphs for each conformer
        batch = self.replicate_graphs_for_conformers(batch)

        # Run MACE on the batch of conformers for each molecule to get MACE features.
        batch = super().forward(batch) 

        # Compute barycenters for each molecule in the batch
        barycenters = self.compute_barycenters(batch)

        return batch, barycenters

    def compute_barycenters(self, batch) -> Batch:
        """
        Computes the barycenters of the conformers in the batch.
        The barycenters are computed using the Fused Gromov-Wasserstein (FGW) method.
        Args:
            batch (torch_geometric.data.Batch): The batch of conformers for each molecule.
        Returns:
            torch_geometric.data.Batch: A new batch with the barycenters.
        """

        # Initialize the list to store the barycenters
        barycenters = []

        for i in range(0, batch.num_graphs, cfg.NUM_CONFORMERS_SAMPLE):
            # Get the batch tensor for the current molecule
            idx = torch.tensor([j for j in range(i, i + cfg.NUM_CONFORMERS_SAMPLE)]).to(cfg.DEVICE)
            mask = torch.isin(batch.batch, idx)

            # Get the conformers for the current molecule
            conformers = batch[i: i + cfg.NUM_CONFORMERS_SAMPLE]

            # Get the MACE features for each node of the conformers and store them in a list of numpy arrays.
            h_mace = batch.h_mace[mask]
            num_nodes = conformers[0].num_nodes
            Ys = [
                h_mace[i: i + num_nodes].detach().cpu().numpy() 
                for i in range(0, num_nodes * cfg.NUM_CONFORMERS_SAMPLE, num_nodes)
            ]

            # Compute the cost matrices for each molecule in the batch, using their edge_index.
            # The cost is the shortest path between the nodes in the graph.
            Cs = self.compute_Cs(conformers)

            # Set the "structure" marginal node probabilities to be uniform for each node.
            ps = [np.ones(len(cs), dtype=np.float32) / len(cs) for cs in Cs]

            # Set the relative weights for each conformer. They show how much each conformer contributes to the barycenter.
            # The weights are set to be uniform for each conformer.
            lambdas = [1. / cfg.NUM_CONFORMERS_SAMPLE for _ in range(cfg.NUM_CONFORMERS_SAMPLE)]

            # Set the size of the barycenter, i.e. the number of nodes in the barycenter graph.
            sizebary = len(Cs[0])

            # Compute the barycenter of the conformers. We get the barycenter node features and discard other outputs,
            # which are the cost matrix and OT plans
            h_bary, _ = fgw_barycenters(
                N=sizebary, 
                Ys=Ys,
                Cs=Cs,
                ps=ps,
                lambdas=lambdas,
            )

            # Convert the barycenter node features to a PyTorch tensor and move it to the same device as the input batch.
            h_bary = torch.tensor(h_bary, dtype=torch.float32).to(cfg.DEVICE)
            barycenter = Data(x=h_bary)

            # Append the barycenter to the list of barycenters
            barycenters.append(barycenter)

        return Batch.from_data_list(barycenters)

    def replicate_graphs_for_conformers(self, batch) -> Batch:
        """
        Replicates each graph in 'batch' for multiple conformers.
        Args:
            batch (torch_geometric.data.Batch): The original batch of graphs.

        Returns:
            torch_geometric.data.Batch: A new batch with each graph replicated.
        """

        new_data_list = []
        for data in batch.to_data_list():
            # Create a new graph for each conformer
            for i in range(cfg.NUM_CONFORMERS_SAMPLE):
                new_data = data.clone()
                new_data.pos = data.pos[:, i, :]
                new_data_list.append(new_data)
        return Batch.from_data_list(new_data_list)
    
    def sample_conformers(self, batch: Batch) -> Batch:
        """
        Randomly samples conformers from the batch.
        The batch should contain multiple conformers for each molecule.
        The conformers are sampled without replacement.
        Args:
            batch (torch_geometric.data.Batch): The original batch of graphs.
        Returns:    
            torch_geometric.data.Batch: A new batch with sampled conformers.
        """

        if batch.pos is not None:
            # Sample conformers
            conformer_idx = np.random.choice(range(cfg.NUM_CONFORMERS), size=cfg.NUM_CONFORMERS_SAMPLE, replace=False)
            batch.pos = batch.pos[:, conformer_idx, :]
            return batch
        else:
            # If no conformers are present, return the batch as is
            raise ValueError("No conformers found in the batch. Please check the input data.")
    
    def compute_Cs(self, conformers) -> list:
        """
        Computes the cost matrices for each molecule in the batch, using their edge_index.
        The cost is the shortest path between the nodes in the graph.
        Args:
            conformers (list): List of conformer graphs.       
        Returns:
            list: List of cost matrices for each conformer.
        """
        
        # Compute the cost matrices for each molecule in the batch, using their edge_index.
        adj_matrices = [to_dense_adj(conformer.edge_index).squeeze() for conformer in conformers]
        Cs_list = [shortest_path(adj.detach().cpu().numpy()) for adj in adj_matrices]
        return Cs_list


