import os
import logging
import numpy as np
import random
import torch

from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg

from models import MolPredictor, BaselineMLP
from data.moldataset import MolDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seeding everything with seed: {seed}")


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)
        predictions = outputs["pred"]
        h_mol = outputs["mol_emb"]
        h_gat = outputs["gat_emb"]
        h_mace = outputs["mace_emb"]

        # Mask NaN values in the target
        # [True, False] if batch size = 1, batch.y = [some_value, NaN]
        batch.y = batch.y
        mask = ~torch.isnan(batch.y)
        masked_predictions = predictions[mask]
        masked_targets = batch.y[mask]

        # Compute loss only on valid targets
        loss = criterion(masked_predictions, masked_targets)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader), h_mol, h_gat, h_mace


def get_next_run_folder(base_dir="runs"):
    """
    Creates the next folder name with consecutive numbers in the given base directory.
    Example: runs/001, runs/002, etc.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing = [name for name in os.listdir(base_dir) if name.isdigit()]
    next_number = max([int(name) for name in existing], default=0) + 1
    return os.path.join(base_dir, f"{next_number:03d}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Seed
    seed_everything()

    # Set device
    device = torch.device(cfg.DEVICE)
    logger.info(f"Using device: {device}")

    # Load dataset
    if cfg.DEBUG:
        logger.info(f"Debug mode: Loading only {cfg.NUM_MOLECULES} molecules for training.")
        NUM_MOLS = cfg.NUM_MOLECULES
        SHUFFLE = False
        BATCH_SIZE = 1
    else:
        logger.info(f"Loading the entire dataset for training.")
        NUM_MOLS = None
        SHUFFLE = True
        BATCH_SIZE = cfg.BATCH_SIZE

    train_dataset = MolDataset(cfg.TRAIN_DIR, scaler_path=cfg.SCALER_PATH)[:NUM_MOLS]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=cfg.NUM_WORKERS)

    # Print data
    if cfg.DEBUG:
        for batch in train_loader:
            logger.info(f"Batch: {batch}")
            logger.info(f"Batch targets: {batch.y}")

    # Initialize model, loss, and optimizer
    if cfg.BASE:
        logger.info("Using Baseline MLP model.")
        model = BaselineMLP().to(device)
    else:
        logger.info("Using MolPredictor model.")
        model = MolPredictor().to(device)

    logger.info(f"Model has {count_parameters(model):,} trainable parameters.")
    logger.info(f"Model: {model}")

    # Define loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    # Define a linear learning rate scheduler
    # def linear_lr_lambda(epoch):
    #    return 1 - epoch / cfg.NUM_EPOCHS

    # scheduler = LambdaLR(optimizer, lr_lambda=linear_lr_lambda)

    # Initialize TensorBoard writer
    run_name = "runs/" + cfg.RUN_NAME if cfg.RUN_NAME else get_next_run_folder()
    logger.info(f"Logging to {run_name} directory")
    writer = SummaryWriter(log_dir=run_name)

    best_loss = float("inf")
    early_stop_counter = 0
    max_early_stop = cfg.MAX_EARLY_STOP

    for epoch in range(cfg.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        train_loss, h_mol, h_gat, h_mace = train(model, train_loader, optimizer, criterion, device)

        # Step the scheduler
        # scheduler.step()

        # Adjust learning rate
        # if epoch < cfg.WARMUP_STEPS:
        #    lr = cfg.LEARNING_RATE * (epoch / cfg.WARMUP_STEPS)
        # else:
        #    lr = cfg.LEARNING_RATE

        # if epoch < cfg.WARMUP_STEPS:
        #    logger.info(f"Warmup phase. Learning rate: {lr:.6f}")
        # else:
        #    logger.info(f"Normal phase. learning rate: {lr:.6f}")

        # for param_group in optimizer.param_groups:
        #    param_group["lr"] = lr

        # Log epoch-level losses
        writer.add_scalar("Train Loss", train_loss, epoch)

        # Log gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Log gradient distributions
                writer.add_histogram(f"gradients/{name}", param.grad, epoch)
                # Log gradient norm
                writer.add_scalar(f"gradient_norms/{name}", param.grad.norm(), epoch)

        # Log embeddings
        all_emb = torch.cat([h_mol, h_gat, h_mace], dim=0)

        # Create labels
        num_labels = h_mol.shape[0]
        labels = ["Mol"] * num_labels + ["GAT"] * num_labels + ["MACE"] * num_labels

        writer.add_embedding(all_emb, metadata=labels, tag="multi_embeddings", global_step=epoch)

        print(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

        # Early stopping logic
        if train_loss < best_loss:
            best_loss = train_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= max_early_stop:
            print(f"Early stopping triggered. No improvement for {cfg.MAX_EARLY_STOP} epochs.")
            break

    # Save the model
    torch.save(model.state_dict(), "models/trained_models/mol_predictor.pth")
    print("Model saved as mol_predictor.pth")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    from evaluation import eval_potency
    from pprint import pprint
    import pickle

    scalers = pickle.load(open(cfg.SCALER_PATH, "rb"))
    scaler_y = scalers["scaler_y"]

    main()

    # Seed
    seed_everything()
    # Load the trained model
    model = MolPredictor().to(cfg.DEVICE)
    model.load_state_dict(torch.load("models/trained_models/mol_predictor.pth"))
    model.eval()

    train_dataset = MolDataset(cfg.TRAIN_DIR, scaler_path=cfg.SCALER_PATH)[: cfg.NUM_MOLECULES]
    shuffle = True if not cfg.DEBUG else False
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.NUM_WORKERS)

    if cfg.DEBUG:
        for batch in train_loader:
            logger.info(f"Batch: {batch}")
            logger.info(f"Batch targets: {batch.y}")

    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}
    predictions = {}
    targets = {}

    for num, key in enumerate(keys):
        predictions[key] = []
        targets[key] = []
        for batch in tqdm(train_loader, desc=f"Evaluating {key}"):
            batch = batch.to(cfg.DEVICE)
            outputs = model(batch)
            pred = outputs["pred"].cpu().detach().numpy()
            pred = scaler_y.inverse_transform(pred)[:, num]
            predictions[key].extend(pred)
            y = scaler_y.inverse_transform(batch.y.cpu().detach().numpy())[:, num]
            targets[key].extend(y)

    pprint(dict(eval_potency(predictions, targets)))
