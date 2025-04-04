import numpy as np
import torch

from models import MACEModel
from torch_geometric.data import Batch

import config as cfg

import sys
sys.path.append('../')

import datamol as dm
import logging
import numpy as np
import os
import pandas as pd 
import torch

from graphium.features import featurizer as gff
from datamol.descriptors.compute import _DEFAULT_PROPERTIES_FN
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose, NormalizeScale
from torch_geometric.utils import remove_self_loops, to_dense_adj
from tqdm import tqdm
from transformers import pipeline



# TODO: Change the number of processed SMILES to the length of the dataframe
# for idx in tqdm(range(10), desc="Processing molecules", total=len(df_data)): -> for idx in tqdm(range(lrn(df_data)), desc="Processing molecules", total=len(df_data)):

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MACEBaryModel(MACEModel):

    def __init__(self, mace_kwargs):
        super().__init__(**mace_kwargs)


    def forward(self, batch):
        """
        Forward pass of the MACE model with barycenter computation.
        This function computes the forward pass of the MACE model and
        additionally computes the barycenter of the node features.
        
        Parameters:
            - batch: Batch of input data that has several molecules.
            - max_iter: Maximum number of iterations for barycenter computation.
            - epsilon: Convergence threshold for barycenter computation.
        """
        # Compute the cost matrices for each molecule in the batch, using their edge_index.
        # The cost is the shortest path between the nodes in the graph.
        Cs_list = self.compute_Cs(batch)

        # Set the probabilites 
        ps = [torch.ones(len(Cs), dtype=torch.float) / len(Cs) for Cs in Cs_list]

        # Sample conformers and replicate graphs for each conformer
        batch = self.sample_conformers(batch)
        batch = self.replicate_graphs_for_conformers(batch)

        # Run MACE to get the node features
        h_mace = super().forward(batch)  

        # Compute the barycenter of the node features
        h_bary = None     


    def forward_3d_bary(self, batch: Batch) -> torch.Tensor:
        """
        Create two embeddings, one for standard 3D aggregation, second for barycenter
            Args:
        z (LongTensor): Atomic number of each atom with shape
            :obj:`[num_atoms]`.
        pos (Tensor): Coordinates of each atom with shape
            :obj:`[num_atoms, 3]`.
        batch (LongTensor, optional): Batch indices assigning each atom to
            a separate molecule with shape :obj:`[num_atoms]`.
            (default: :obj:`None`)
        data_batch (Batch, optional): Batch object with additional data, such as covalent bonds attributes
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h_shared = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h_shared = h_shared + interaction(h_shared, edge_index, edge_weight, edge_attr)

        h = self.lin1(h_shared)
        h = self.lin2(h)
        h = self.act(h)

        h_bary = self.lin1_bary(h_shared)
        h_bary = self.lin2_bary(h_bary)
        h_bary = self.act(h_bary)
        return h, h_bary

    def forward_w_barycenter(self, batch: Batch, num_conformers: int, max_iter: int = 100, epsilon: float = 0.1) -> torch.Tensor:
        """
        Forward pass of the MACE model with barycenter computation.
        This function computes the forward pass of the MACE model and
        additionally computes the barycenter of the node features.
        
        Parameters:
            - batch: Batch of input data that has several molecules.
            - num_conformers: Number of conformers per molecule.
            - max_iter: Maximum number of iterations for barycenter computation.
            - epsilon: Convergence threshold for barycenter computation.
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # h = self.embedding(z)
        h_3d, h_bary = self.forward_3d_bary(z, pos, batch)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        # edge_attr = self.distance_expansion(edge_weight)

        batch_size = int(len(batch.unique()) / num_conformers)
        h_3d_non, h_bary = self._compute_barycenter(
            node_feature=h_bary,
            edge_index=edge_index,
            batch=batch,
            batch_size=batch_size,
            num_conformers=num_conformers,
        )
        h_3d = self.readout(h_3d, batch, dim=0)
        return h_3d, h_bary
    
    def replicate_graphs_for_conformers(self, batch):
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
        # Sample conformers
        conformer_idx = np.random.choice(range(cfg.NUM_CONFORMERS), size=cfg.NUM_CONFORMERS_SAMPLE, replace=False)
        batch.pos = batch.pos[:, conformer_idx, :]
        return batch
    
    def compute_Cs(self, batch):
        adj_matrices = to_dense_adj(batch.edge_index, batch=batch.batch)
        adj_matrices = [adj_matrices[i] for i in range(batch.num_graphs)]
        Cs_list = [shortest_path(adj.detach().cpu().numpy()) for adj in adj_matrices]
        return [torch.tensor(Cs, dtype=torch.float).to(cfg.DEVICE) for Cs in Cs_list]

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
    

class MolDataset(InMemoryDataset):
    def __init__(
            self, 
            root: str, 
            pre_transform: callable = Compose([NormalizeScale(), ConcatenateGlobal()]), 
            transform = None, 
            **kwargs
        ):
        super().__init__(root, pre_transform=pre_transform, transform=transform, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        raw_files = os.listdir(self.raw_dir)
        return raw_files
    
    @property
    def processed_file_names(self) -> list[str]:
        raw_files = os.listdir(self.raw_dir)
        processed_files = [raw_file.replace(".csv", ".pt") for raw_file in raw_files]
        return processed_files
    
    def normalize_mol_descriptors(self, data_list, desc_mins, desc_maxs):
        # Normalize the molecular descriptors to the range [0, 1]
        for data in data_list:
            data.u_dm = (data.u_dm - desc_mins) / (desc_maxs - desc_mins)
            data.u_dm = torch.nan_to_num(data.u_dm, nan=0.0, posinf=0.0, neginf=0.0)

    def process(self) -> None:
        raw_files = self.raw_file_names
        
        # Get ChemBERTa model
        pipe = pipeline("feature-extraction", model="seyonec/ChemBERTa-zinc-base-v1")
        
        # Read the raw data
        df_data = pd.read_csv(os.path.join(self.raw_dir, raw_files[0]), header=0)

        # Create a list to store the data objects
        data_list = []

        # Init an array to hold the statistics of the molecular descriptors
        desc_mins = torch.ones(len(_DEFAULT_PROPERTIES_FN), dtype=torch.float) * 10_000
        desc_maxs = -torch.ones(len(_DEFAULT_PROPERTIES_FN), dtype=torch.float) * 10_000

        for idx in tqdm(range(10), desc="Processing molecules", total=len(df_data)):
            dm.disable_rdkit_log() # stop logging a lot of info for datamol methods calls
            
            smile = df_data['CXSMILES'][idx]
            mol = dm.to_mol(smile, add_hs=True)
            mol = dm.sanitize_mol(mol)
            mol = dm.fix_mol(mol)
            mol = dm.standardize_mol(
                mol,
                disconnect_metals=True,
                normalize=True,
                reionize=True,
                uncharge=False,
                stereo=True,
            )

            # Get CHEMBERTa features              
            u = pipe(df_data['CXSMILES'].iloc[idx])
            u_chem = torch.tensor(u[0][0], dtype=torch.float)
            
            # Get Datamol molecular features
            descriptors = dm.descriptors.compute_many_descriptors(mol)
            u = list(descriptors.values())
            u_dm = torch.tensor(u, dtype=torch.float)

            # Update the statistics of the molecular descriptors
            desc_maxs= torch.where(u_dm > desc_maxs, u_dm, desc_maxs)
            desc_mins = torch.where(u_dm < desc_mins, u_dm, desc_mins)
            
            # Allowable atomic node and edge features
            atomic_features = [ "atomic-number", "mass", "weight","valence","total-valence",
                                "implicit-valence","hybridization","ring", "in-ring","min-ring",
                                "max-ring","num-ring","degree","radical-electron","formal-charge",
                                "vdw-radius","covalent-radius","electronegativity","ionization",
                                "first-ionization","metal","single-bond","aromatic-bond",
                                "double-bond","triple-bond","is-carbon","group","period" ]
            
            edge_features = ["bond-type-onehot", "in-ring", "conjugated", "estimated-bond-length"]
            
            # Get float atomic features
            values_atomic_feat = gff.get_mol_atomic_features_float(mol, atomic_features, mask_nan='warn').values()
            x_array = np.column_stack(list(values_atomic_feat))
            x = torch.tensor(x_array, dtype=torch.float)
            
            # Get one-hot atomic numbers
            atoms_onehot = gff.get_mol_atomic_features_onehot(mol, ["atomic-number"]).values()
            atoms_onehot = np.column_stack(list(atoms_onehot))
            atoms_onehot = torch.tensor(atoms_onehot, dtype=torch.float)
            # Transform onehot to indices of possible atoms
            atoms = torch.where(atoms_onehot > 0)[1]

            # Generate conformers
            try:
                mol_confs = dm.conformers.generate(mol, n_confs=cfg.NUM_CONFORMERS)
                list_xyz = [mol_confs.GetConformer(i).GetPositions() for i in range(cfg.NUM_CONFORMERS)]
                pos_array = np.stack(list_xyz, axis=1)
            except Exception as e:
                logger.warning(f"Conformer generation failed for {smile}: {e}")
                logger.info("Setting atom positions to all zeros")
                pos_array = np.zeros((x.shape[0], cfg.NUM_CONFORMERS, 3))
            pos = torch.tensor(pos_array, dtype=torch.float)
            
            # Additional edge features:"bond-type-onehot", "stereo",conformer-bond-length" (might cause problems with complex molecules)
            edge_dict = gff.get_mol_edge_features(mol, edge_features, mask_nan='warn')          
            edge_list = list(edge_dict.values())
            edge_attr = np.column_stack(edge_list)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Repeat edge_attr twice to account for both directions of the edges
            edge_attr = edge_attr.repeat_interleave(2, dim=0)

            # Get adjacency matrix
            adj = gff.mol_to_adjacency_matrix(mol)
            edge_index = torch.stack([torch.tensor(adj.coords[0], dtype=torch.int64), torch.tensor(adj.coords[1], dtype=torch.int64)], dim=0)

            # Get the target values
            df_y = df_data[["pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)"]].iloc[idx]
            y = torch.tensor(np.array(df_y), dtype=torch.float)
            
            # Get a PyG data object
            data = Data(u_chem=u_chem, u_dm=u_dm, edge_attr=edge_attr, pos=pos, x=x, y=y, atoms=atoms, edge_index=edge_index)
            
            # Append the data object to the list
            data_list.append(data)
        

        # Normalize molecular descriptors
        self.normalize_mol_descriptors(data_list, desc_mins, desc_maxs)

        # Apply the pre_transform if provided
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        #Save the processed data
        self.save(data_list, self.processed_paths[0])

if __name__ == "__main__":
    
    mace_kwargs = {
        "r_max": 5,
        "num_bessel": 10,
        "num_polynomial_cutoff": 6,
        "max_ell": 2,
        "correlation": 3,
        "num_layers": 5,
        "emb_dim": 64,
        "hidden_irreps": None,
        "mlp_dim": 256,
        "in_dim": cfg.NUM_POSSIBLE_ATOMS,
        "out_dim": 1,
        "aggr": "sum",
        "pool": "sum",
        "batch_norm": True,
        "residual": True,
        "equivariant_pred": True,
        "as_featurizer": True,
    }

    model = MACEBaryModel(mace_kwargs)
    train_dataset = MolDataset(root=cfg.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
    )

    
    batch = next(iter(train_loader))
    batch = batch.to(cfg.DEVICE)
    mace_bary = model.to(cfg.DEVICE)

    # Run on batch
    mace_bary.eval()
    mace_bary(batch)



