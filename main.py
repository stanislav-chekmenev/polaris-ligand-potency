import os
import logging
import torch

from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg

from models.mol_predictor import MolPredictor
from data.moldataset import MolDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader), desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch)

        # Mask NaN values in the target
        # [True, False] if batch size = 1, batch.y = [some_value, NaN]
        batch.y = batch.y.view(-1, cfg.PREDICTION_DIM)
        mask = ~torch.isnan(batch.y)
        masked_predictions = predictions[mask]
        masked_targets = batch.y[mask]

        # Compute loss only on valid targets
        loss = criterion(masked_predictions, masked_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def get_next_run_folder(base_dir="runs"):
    """
    Creates the next folder name with consecutive numbers in the given base directory.
    Example: runs/001, runs/002, etc.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing = [name for name in os.listdir(base_dir) if name.isdigit()]
    next_number = max([int(name) for name in existing], default=0) + 1
    return os.path.join(base_dir, f"{next_number:03d}")


def main():
    # Set device
    device = torch.device(cfg.DEVICE)

    # Load dataset
    train_dataset = MolDataset(cfg.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    # Initialize model, loss, and optimizer
    model = MolPredictor().to(device)
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=get_next_run_folder())

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)

        # Log epoch-level losses
        writer.add_scalar("Train Loss", train_loss, epoch)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "models/trained_models/mol_predictor.pth")
    print("Model saved as mol_predictor.pth")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    from evaluation import eval_potency

    # main()

    # Load the trained model
    model = MolPredictor().to(cfg.DEVICE)
    model.load_state_dict(torch.load("models/trained_models/mol_predictor.pth"))

    train_dataset = MolDataset(cfg.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}
    predictions = {}
    targets = {}

    for num, key in enumerate(keys):
        predictions[key] = []
        targets[key] = []
        for batch in tqdm(train_loader, desc=f"Evaluating {key}"):
            batch = batch.to(cfg.DEVICE)
            pred = model(batch).cpu().detach().numpy()[num]
            predictions[key].extend(pred)
            y = batch.y.cpu().detach().numpy().reshape(-1, 2)[num]
            targets[key].extend(y)

    print(eval_potency(predictions, targets))
