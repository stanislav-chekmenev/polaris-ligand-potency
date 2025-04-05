import torch 
import multiprocessing

from pathlib import Path

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


#### DATA CONFIG ####
ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"


#### GAT CONFIG ####
EDGE_DIM = 8