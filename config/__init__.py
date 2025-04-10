import torch
import multiprocessing

from pathlib import Path
from torch_geometric.nn.pool import global_add_pool, global_mean_pool

#### TRAINING CONFIG ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
MAX_EARLY_STOP = 50
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 50
RUN_NAME = None  # "mp_mace_LR_1e-3_CLIP_5_mean_pool_BS_1_CONF_1_no_sample"
GRADIENT_CLIP = 5.0

#### DEBUGGING CONFIG ####
NUM_MOLECULES = 4
DEBUG = True

#### BASELINE CONFIG ####
BASE = True

# CONFORMERS
NUM_CONFORMERS = 10
NUM_CONFORMERS_SAMPLE = 1
NUM_THREADS = 8

# DATA LOADING
BATCH_SIZE = 2
NUM_WORKERS = multiprocessing.cpu_count() - 1

# FEATURES
NUM_POSSIBLE_ATOMS = 44  # Do not change, since it's the number of possible atoms given by Graphium
PREDICTION_DIM = 2  # Do not change, since it's the number of possible properties to predict

# FEATURE EMBEDDER #
IN_MOL_DIM = 790
NODE_DIM = 18
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
    "pool": "mean",
    "batch_norm": True,
    "residual": True,
    "equivariant_pred": False,
    "as_featurizer": True,
}

# GAT
EDGE_DIM = 8
IN_GAT_DIM = EMB_DIM
GAT_NODE_AGGREGATION = global_mean_pool

# NODE FEATURE AGGREGATION
NODE_AGGREGATION = global_mean_pool

#### DATA CONFIG ####
ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"
TEST_DIR = ROOT / "data" / "test"
SCALER_PATH = ROOT / "data" / "train" / "processed" / "scalers.pkl"

# MODELS DIR
MODELS_DIR = ROOT / "models" / "trained_models"
