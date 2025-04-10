import datamol as dm
import logging
import numpy as np
import os
import pandas as pd
import pickle
import torch

from datamol.descriptors.compute import _DEFAULT_PROPERTIES_FN
from graphium.features import featurizer as gff
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose
from transformers import pipeline
from tqdm import tqdm

import config as cfg
from data.transforms import ConcatenateGlobal, NormalizeScaleWithZeros


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        scaler_path: str,
        pre_transform: callable = Compose([NormalizeScaleWithZeros(), ConcatenateGlobal()]),
        transform=None,
        **kwargs,
    ):
        self.scaler_path = scaler_path
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

    def apply_scalers(self, data, scaler_x, scaler_y, scaler_u, scaler_edge):
        # Apply the scalers to the data
        data.x = torch.tensor(scaler_x.transform(data.x), dtype=torch.float)
        data.y = torch.tensor(scaler_y.transform(data.y), dtype=torch.float)
        data.u = torch.tensor(scaler_u.transform(data.u), dtype=torch.float)
        data.edge_attr[:, -1] = torch.tensor(
            scaler_edge.transform(data.edge_attr[:, -1].view(-1, 1)), dtype=torch.float
        ).view(-1)
        return data

    def process(self) -> None:
        raw_files = self.raw_file_names

        # Get ChemBERTa model
        pipe = pipeline("feature-extraction", model="seyonec/ChemBERTa-zinc-base-v1", device="cpu")

        # Read the raw data
        df_data = pd.read_csv(os.path.join(self.raw_dir, raw_files[0]))

        # Create a list to store the data objects
        data_list = []

        for idx in tqdm(range(len(df_data)), desc="Processing molecules", total=len(df_data)):
            dm.disable_rdkit_log()  # stop logging a lot of info for datamol methods calls

            smile = df_data["CXSMILES"][idx]
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
            u = pipe(df_data["CXSMILES"].iloc[idx])
            u_chem = torch.tensor(u[0][0], dtype=torch.float)

            # Get Datamol molecular features
            descriptors = dm.descriptors.compute_many_descriptors(mol)
            u = list(descriptors.values())
            u_dm = torch.tensor(u, dtype=torch.float)

            # Allowable atomic node and edge features
            atomic_features = [
                "atomic-number",
                "mass",
                "weight",
                "valence",
                "total-valence",
                "implicit-valence",
                "hybridization",
                "ring",
                "min-ring",
                "max-ring",
                "num-ring",
                "degree",
                "radical-electron",
                "formal-charge",
                "vdw-radius",
                "covalent-radius",
                "electronegativity",
                "ionization",
                "first-ionization",
                "group",
                "period",
            ]

            edge_features = ["bond-type-onehot", "in-ring", "conjugated", "estimated-bond-length"]

            # Get float atomic features
            values_atomic_feat = gff.get_mol_atomic_features_float(mol, atomic_features, mask_nan="warn").values()
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
                mol_confs = dm.conformers.generate(
                    mol,
                    n_confs=cfg.NUM_CONFORMERS,
                    num_threads=cfg.NUM_THREADS,
                    minimize_energy=True,
                    energy_iterations=200,
                )
                list_xyz = [mol_confs.GetConformer(i).GetPositions() for i in range(cfg.NUM_CONFORMERS)]
                pos_array = np.stack(list_xyz, axis=1)
            except Exception as e:
                logger.warning(f"Conformer generation failed for {smile}: {e}")
                logger.info("Setting all positions to zeros")
                pos_array = np.zeros((x.shape[0], cfg.NUM_CONFORMERS, 3))

            pos = torch.tensor(pos_array, dtype=torch.float)

            # Additional edge features:"bond-type-onehot", "stereo",conformer-bond-length" (might cause problems with complex molecules)
            edge_dict = gff.get_mol_edge_features(mol, edge_features, mask_nan="warn")
            edge_list = list(edge_dict.values())
            edge_attr = np.column_stack(edge_list)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            # Repeat edge_attr twice to account for both directions of the edges
            edge_attr = edge_attr.repeat_interleave(2, dim=0)

            # Get adjacency matrix
            adj = gff.mol_to_adjacency_matrix(mol)
            edge_index = torch.stack(
                [torch.tensor(adj.coords[0], dtype=torch.int64), torch.tensor(adj.coords[1], dtype=torch.int64)], dim=0
            )

            # Get the target values
            df_y = df_data[["pIC50 (MERS-CoV Mpro)", "pIC50 (SARS-CoV-2 Mpro)"]].iloc[idx]
            y = torch.tensor(np.array(df_y), dtype=torch.float).view(-1, cfg.PREDICTION_DIM)

            # Get a PyG data object
            data = Data(
                u_chem=u_chem, u_dm=u_dm, edge_attr=edge_attr, pos=pos, x=x, y=y, atoms=atoms, edge_index=edge_index
            )

            # Append the data object to the list
            data_list.append(data)

        # Normalize molecular descriptors
        # self.normalize_mol_descriptors(data_list, desc_mins, desc_maxs)

        # Apply the pre_transform if provided
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Scale features
        if "train" in self.raw_dir:
            # Initialize the scalers
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            scaler_u = StandardScaler()
            scaler_edge = StandardScaler()

            # Fit the scalers on the training data
            x = torch.cat([data.x for data in data_list], dim=0)
            y = torch.cat([data.y for data in data_list], dim=0)
            u = torch.cat([data.u for data in data_list], dim=0)
            edge_attr = torch.cat([data.edge_attr[:, -1] for data in data_list], dim=0)

            scaler_x = scaler_x.fit(x)
            scaler_y = scaler_y.fit(y)
            scaler_u = scaler_u.fit(u)
            scaler_edge = scaler_edge.fit(edge_attr.view(-1, 1))

            # Apply the scalers
            data_list = [self.apply_scalers(data, scaler_x, scaler_y, scaler_u, scaler_edge) for data in data_list]

            # Save the scalers
            scalers = {"scaler_x": scaler_x, "scaler_y": scaler_y, "scaler_u": scaler_u, "scaler_edge": scaler_edge}
            pickle.dump(scalers, open(self.scaler_path, "wb"))

        else:
            # Load the scalers
            scalers = pickle.load(open(self.scaler_path, "rb"))
            scaler_x = scalers["scaler_x"]
            scaler_y = scalers["scaler_y"]
            scaler_u = scalers["scaler_u"]
            scaler_edge = scalers["scaler_edge"]

            # Apply the scalers
            data_list = [self.apply_scalers(data, scaler_x, scaler_y, scaler_u, scaler_edge) for data in data_list]

        # Save the processed data
        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    dataset = MolDataset(root=cfg.TRAIN_DIR, scaler_path=cfg.SCALER_PATH)
    dataset = MolDataset(root=cfg.TEST_DIR, scaler_path=cfg.SCALER_PATH)
