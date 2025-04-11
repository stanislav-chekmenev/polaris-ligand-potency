import os
import logging
import numpy as np
import pickle
import random
import torch

from pprint import pprint
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config as cfg

from evaluation import eval_potency
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


def train(model, loader, optimizer, criterion, device, current_batch, warmup_batches):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, total=len(loader), desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Adjust learning rate during warm-up
        if current_batch <= warmup_batches:
            lr = cfg.LEARNING_RATE * (current_batch / warmup_batches)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # Log the learning rate
        current_batch += 1

        # Forward pass
        if cfg.BASE:
            predictions = model(batch)
        else:
            outputs = model(batch)
            predictions = outputs["pred"]
            h_mol = outputs["mol_emb"]
            h_gat = outputs["gat_emb"]
            h_mace = outputs["mace_emb"]

        # Mask NaN values in the target
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

    result = {
        "loss": total_loss / len(loader),
        "h_mol": h_mol if not cfg.BASE else None,
        "h_gat": h_gat if not cfg.BASE else None,
        "h_mace": h_mace if not cfg.BASE else None,
        "current_batch": current_batch,
    }

    return result


def val(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Running validation"):
            batch = batch.to(device)

            if cfg.BASE:
                predictions = model(batch)
            else:
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
            total_loss += loss.item()

    result = {
        "loss": total_loss / len(loader),
        "h_mol": h_mol if not cfg.BASE else None,
        "h_gat": h_gat if not cfg.BASE else None,
        "h_mace": h_mace if not cfg.BASE else None,
    }

    return result


def get_next_run_folder(base_dir="runs", evaluation=False):
    """
    Creates the next folder name with consecutive numbers in the given base directory.
    Example: 001, 002, etc.
    """
    if evaluation:
        base_dir = os.path.join(base_dir, "evaluation")
        if cfg.DEBUG:
            base_dir = os.path.join(base_dir, "debug")
        else:
            base_dir = os.path.join(base_dir, "final")
    else:
        suffix = "debug" if cfg.DEBUG else "train"
        base_dir = os.path.join(base_dir, suffix)
    os.makedirs(base_dir, exist_ok=True)
    existing = [name for name in os.listdir(base_dir) if name.isdigit()]
    next_number = max([int(name) for name in existing], default=0) + 1
    return f"{next_number:03d}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_name():
    full_model_name = "mol_predictor" if not cfg.MODEL_NAME else cfg.MODEL_NAME
    full_model_name += ".pth"
    model_name = "baseline_mlp.pth" if cfg.BASE else full_model_name
    return model_name


def main():
    # Seed
    seed_everything()

    # Set device
    device = torch.device(cfg.DEVICE)
    logger.info(f"Using device: {device}")

    # Load dataset
    if cfg.DEBUG:
        logger.info(f"Debug mode: Loading only {cfg.NUM_MOLECULES} molecules for training.")
        NUM_MOLECULES = cfg.NUM_MOLECULES
        SHUFFLE = False
        BATCH_SIZE = 4
    else:
        logger.info(f"Loading the entire dataset for training.")
        NUM_MOLECULES = None
        SHUFFLE = True
        BATCH_SIZE = cfg.BATCH_SIZE

    train_dataset = MolDataset(cfg.TRAIN_DIR, scaler_path=cfg.SCALER_PATH)[:NUM_MOLECULES]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=cfg.NUM_WORKERS)

    if cfg.DEBUG:
        # Print batch info
        for batch in train_loader:
            logger.info(f"First 5 batch positions of conformer 0: \n {batch.pos[:5, 0, :]}")
            logger.info(f"Batch targets: {batch.y}")

    if not cfg.DEBUG:
        val_dataset = MolDataset(cfg.VAL_DIR, scaler_path=cfg.SCALER_PATH)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)

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

    # Initialize LinearLR scheduler
    def lr_lambda(epoch):
        # Ratio of final LR to initial LR
        ratio = cfg.FINAL_LEARNING_RATE / cfg.LEARNING_RATE

        if cfg.WARMUP_BATCHES and epoch < cfg.ANNEALING_STEPS + len(train_loader) // cfg.WARMUP_BATCHES:
            fraction = epoch / cfg.ANNEALING_STEPS
            if fraction > 1.0:
                fraction = 1.0
            return (1.0 - fraction) + fraction * ratio
        else:
            return ratio

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Initialize TensorBoard writer
    sub_dir = "debug" if cfg.DEBUG else "train"
    run_name = cfg.RUN_NAME if cfg.RUN_NAME else get_next_run_folder()
    log_dir = os.path.join("runs", sub_dir, run_name)
    logger.info(f"Logging to {log_dir} directory")
    writer = SummaryWriter(log_dir=log_dir)

    # Log config data to TensorBoard
    config_data = {k: str(v) for k, v in cfg.__dict__.items() if k.isupper()}
    writer.add_hparams(config_data, {"placeholder": 0})

    # Warm-up configuration
    warmup_batches = cfg.WARMUP_BATCHES
    current_batch = 1
    warmup_done = False

    best_loss = float("inf")
    early_stop_counter = 0
    max_early_stop = cfg.MAX_EARLY_STOP

    for epoch in range(cfg.NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        results_train = train(model, train_loader, optimizer, criterion, device, current_batch, warmup_batches)
        current_batch = results_train["current_batch"]

        logger.info(f"Model's device: {next(model.parameters()).device}")
        results_val = not cfg.DEBUG and val(model, val_loader, criterion, device)

        # Step the scheduler if the warm-up done
        if warmup_done:
            scheduler.step()

        # Warmup done?
        if not warmup_done and current_batch > warmup_batches:
            logger.info(f"Warmup finished.")
            warmup_done = True

        # Log epoch-level losses
        logger.info(f"Train Loss: {results_train['loss']:.5f}")
        not cfg.DEBUG and logger.info(f"Validation Loss: {results_val['loss']:.5f}")

        # Log losses to TensorBoard
        writer.add_scalar("Train Loss", results_train["loss"], epoch)
        not cfg.DEBUG and writer.add_scalar("Val Loss", results_val["loss"], epoch)

        # Log learning rate
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        # Log gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Log gradient distributions
                writer.add_histogram(f"gradients/{name}", param.grad, epoch)
                # Log gradient norm
                writer.add_scalar(f"gradient_norms/{name}", param.grad.norm(), epoch)

        # Log embeddings
        if not cfg.BASE:
            h_mol = results_train["h_mol"]
            h_gat = results_train["h_gat"]
            h_mace = results_train["h_mace"]

            # Concatenate embeddings
            all_emb = torch.cat([h_mol, h_gat, h_mace], dim=0)

            # Create labels
            num_labels = h_mol.shape[0]
            labels = ["Mol"] * num_labels + ["GAT"] * num_labels + ["MACE"] * num_labels

            # Write embeddings
            writer.add_embedding(all_emb, metadata=labels, tag="multi_embeddings", global_step=epoch)

        # Early stopping logic
        if not cfg.DEBUG:
            if results_val["loss"] < best_loss:
                best_loss = results_val["loss"]
                early_stop_counter = 0

                # Log the best loss
                logger.info(f"Best validation loss: {best_loss:.5f}. Saving model...")

                # Save model
                if not os.path.exists(cfg.MODELS_DIR):
                    os.makedirs(cfg.MODELS_DIR)

                torch.save(model.state_dict(), os.path.join(cfg.MODELS_DIR, get_model_name()))
                logger.info(f"Model saved as {get_model_name()}")
            else:
                if cfg.MAX_EARLY_STOP:
                    early_stop_counter += 1
        else:
            if results_train["loss"] < best_loss:
                best_loss = results_train["loss"]
                early_stop_counter = 0

                # Log the best loss
                logger.info(f"Best train loss: {best_loss:.5f}. Saving model...")

                # Save model
                if not os.path.exists(cfg.MODELS_DIR):
                    os.makedirs(cfg.MODELS_DIR)

                torch.save(model.state_dict(), os.path.join(cfg.MODELS_DIR, get_model_name()))
                logger.info(f"Model saved as {get_model_name()}")
            else:
                early_stop_counter += 1

        if cfg.MAX_EARLY_STOP and early_stop_counter >= max_early_stop:
            print(f"Early stopping triggered. No improvement for {cfg.MAX_EARLY_STOP} epochs.")
            break

    # Close the TensorBoard writer
    writer.close()


def evaluate():
    # Evaluate the model
    seed_everything()

    # Load scalers
    scalers = pickle.load(open(cfg.SCALER_PATH, "rb"))
    scaler_y = scalers["scaler_y"]

    # Load the trained model
    if cfg.BASE:
        logger.info("Using Baseline MLP model.")
        model = BaselineMLP().to(cfg.DEVICE)
    else:
        logger.info("Using MolPredictor model.")
        model = MolPredictor().to(cfg.DEVICE)

    model.load_state_dict(torch.load(os.path.join(cfg.MODELS_DIR, get_model_name())))
    model.eval()

    # Load the dataset
    if cfg.DEBUG:
        ROOT = cfg.TRAIN_DIR
        NUM_MOLECULES = cfg.NUM_MOLECULES
        BATCH_SIZE = 4
    else:
        ROOT = cfg.TEST_DIR
        NUM_MOLECULES = None
        BATCH_SIZE = cfg.BATCH_SIZE

    logger.info(f"Loading dataset from {ROOT} with {NUM_MOLECULES} molecules.")
    eval_dataset = MolDataset(ROOT, scaler_path=cfg.SCALER_PATH)[:NUM_MOLECULES]
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    if cfg.DEBUG:
        # Print batch info
        for batch in eval_loader:
            logger.info(f"First 5 batch positions of conformer 0: \n {batch.pos[:5, 0, :]}")
            logger.info(f"Batch targets: {batch.y}")

    # Evaluate the model
    predictions = {}
    targets = {}
    keys = {"pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"}

    for num, key in enumerate(keys):
        predictions[key] = []
        targets[key] = []
        for batch in tqdm(eval_loader, desc=f"Evaluating {key}"):
            batch = batch.to(cfg.DEVICE)

            # Forward pass
            if cfg.BASE:
                pred = model(batch)
            else:
                pred = model(batch)["pred"]

            if cfg.DEBUG:
                logger.info(f"Batch targets: {batch.y}")
                logger.info(f"Batch predictions: {pred}")

            pred = pred.cpu().detach().numpy()
            pred = scaler_y.inverse_transform(pred)[:, num]
            y = scaler_y.inverse_transform(batch.y.cpu().detach().numpy())[:, num]

            # Add current pred to predictions and y to targets
            predictions[key].extend(pred)
            targets[key].extend(y)

    # Run evaluation script (NaNs are taken into account inside the function)
    logger.info("Running evaluation script...")
    evaluation_results = dict(eval_potency(predictions, targets))

    # Log evaluation results to TensorBoard
    sub_dir = "debug" if cfg.DEBUG else "final"
    run_name = cfg.RUN_NAME if cfg.RUN_NAME else get_next_run_folder()
    log_dir = os.path.join("runs", "evaluation", sub_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Helper function to create a Markdown table
    def create_markdown_table(results_dict):
        table_header = "| Metric | Value |\n|--------|-------|\n"
        table_rows = "\n".join([f"| {key} | {value:.6f} |" for key, value in results_dict.items()])
        return table_header + table_rows

    # Log aggregated results as a table
    if "aggregated" in evaluation_results:
        aggregated_table = create_markdown_table(evaluation_results["aggregated"])
        writer.add_text("Evaluation Results/Aggregated", aggregated_table, global_step=0)

    # Log individual results for each key
    for key in ["pIC50 (SARS-CoV-2 Mpro)", "pIC50 (MERS-CoV Mpro)"]:
        if key in evaluation_results:
            key_table = create_markdown_table(evaluation_results[key])
            writer.add_text(f"Evaluation Results/{key}", key_table, global_step=0)

    logger.info("Evaluation results:")
    pprint(evaluation_results)

    writer.close()


if __name__ == "__main__":
    # Train the model
    main()

    # Evaluate the model
    evaluate()
