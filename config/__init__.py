import torch
import multiprocessing

from pathlib import Path
from torch_geometric.nn.pool import global_mean_pool

#### TRAINING CONFIG ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 50
EMBEDDER_LEARNING_RATE = 5e-5
FINAL_EMBEDDER_LEARNING_RATE = 1e-5
LEARNING_RATE = 1e-3
ANNEALING_STEPS = 25
FINAL_LEARNING_RATE = 1e-4
MAX_EARLY_STOP = 10
WEIGHT_DECAY = 1e-5
WARMUP_BATCHES = 500
RUN_NAME = "MOL_PRED_WITH_ATT_EMB_DIM_32_BS_4_LRO_1e-3_1e-4_LRE_5e-5_1e-5"
GRADIENT_CLIP = 1.0
MODEL_NAME = "mol_predictor_att_diff_lrs"

#### DEBUGGING CONFIG ####
NUM_MOLECULES = 4
DEBUG = False
DEBUG_ONLY_ONE_CONFORMER = False

#### BASELINE CONFIG ####
BASE = False

# CONFORMERS
NUM_CONFORMERS = 10
NUM_CONFORMERS_SAMPLE = 5
NUM_THREADS = 8

# DATA LOADING
BATCH_SIZE = 4
NUM_WORKERS = multiprocessing.cpu_count() // 2

# FEATURES
NUM_POSSIBLE_ATOMS = 44  # Do not change, since it's the number of possible atoms given by Graphium
PREDICTION_DIM = 2  # Do not change, since it's the number of possible properties to predict

# FEATURE EMBEDDER #
IN_MOL_DIM = 790
NODE_DIM = 18
EMB_DIM = 32

# ATTENTION
NUM_HEADS = 4
IN_ATTENTION_DIM = EMB_DIM

# MACE BARYCENTER
MACE_KWARGS = {
    "r_max": 10.0,
    "num_bessel": 8,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "correlation": 3,
    "num_layers": 3,
    "emb_dim": EMB_DIM,
    "hidden_irreps": None,
    "mlp_dim": 32,
    "in_dim": NUM_POSSIBLE_ATOMS,
    "out_dim": 1,
    "aggr": "sum",
    "pool": "sum",
    "batch_norm": False,
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
SUFFIX = "debug" if DEBUG else "final"
MODELS_DIR = ROOT / "models" / "trained_models" / SUFFIX
