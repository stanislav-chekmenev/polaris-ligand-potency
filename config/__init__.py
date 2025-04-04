import torch 
import multiprocessing

from pathlib import Path

#### TRAINING CONFIG ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CONFORMERS = 10
NUM_CONFORMERS_SAMPLE = 3
BATCH_SIZE = 1
NUM_WORKERS = multiprocessing.cpu_count() - 1

#### DATA CONFIG ####
ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
NUM_POSSIBLE_ATOMS = 44

#### GAT CONFIG ####
EDGE_DIM = 8