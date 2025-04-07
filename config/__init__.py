import torch 
import multiprocessing

from pathlib import Path
from torch_geometric.nn.pool import global_add_pool

#### TRAINING CONFIG ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFORMERS
NUM_CONFORMERS = 10
NUM_CONFORMERS_SAMPLE = 3

# DATA LOADING
BATCH_SIZE = 2
NUM_WORKERS = multiprocessing.cpu_count() - 1

# FEATURES
NUM_POSSIBLE_ATOMS = 44 # Do not change, since it's the number of possible atoms given by Graphium
PREDICTION_DIM = 2 # Do not change, since it's the number of possible properties to predict

# FEATURE EMBEDDER #
IN_MOL_DIM = 790
NODE_DIM = 24
EMB_DIM = 64

# ATTENTION
NUM_HEADS = 4
IN_ATTENTION_DIM = EMB_DIM

# MACE BARYCENTER 
MACE_KWARGS = MACE_KWARGS = {
    "r_max": 10.0,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "correlation": 3,
    "num_layers": 5,
    "emb_dim": EMB_DIM,
    "hidden_irreps": None,
    "mlp_dim": 64,
    "in_dim": NUM_POSSIBLE_ATOMS,
    "out_dim": 1,
    "aggr": "sum",
    "pool": "sum",
    "batch_norm": True,
    "residual": True,
    "equivariant_pred": False,
    "as_featurizer": True
}

# GAT
EDGE_DIM = 8
IN_GAT_DIM = EMB_DIM
GAT_NODE_AGGREGATION = global_add_pool

# NODE FEATURE AGGREGATION
NODE_AGGREGATION = global_add_pool


#### DATA CONFIG ####
ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
