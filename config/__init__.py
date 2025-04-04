import torch 
import multiprocessing

from pathlib import Path

#### TRAINING CONFIG ####
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CONFORMERS = 5
BATCH_SIZE = 2
NUM_WORKERS = multiprocessing.cpu_count() - 1

#### DATA CONFIG ####
ROOT = Path(__file__).parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"

#### GAT CONFIG ####
EDGE_DIM = 8