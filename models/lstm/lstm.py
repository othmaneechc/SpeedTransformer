# lstm.py

import os

# Set the CUBLAS_WORKSPACE_CONFIG environment variable before importing PyTorch
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

import sys
import argparse
import logging
import torch
import torch.nn as nn
import joblib
import random
import numpy as np

from data_utils import DataHandler
from models import LSTMTripClassifier
from trainer import Trainer

def setup_logger(log_file='lstm.log'):
    """
    Set up a logger that writes logs to a file and to the console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicating handlers if they already exist
    if logger.hasHandlers():
        return logger

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms
    torch.use_deterministic_algorithms(True)

def main(
    data_path,
    feature_columns,
    target_column,
    traj_id_column='traj_id',
    batch_size=1024,
    num_workers=4,
    learning_rate=0.0005,
    weight_decay=1e-4,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    num_epochs=50,
    patience=7,
    max_grad_norm=5.0,
    checkpoint_dir='.',
    scaler_path='scaler.joblib',
    label_encoder_path='label_encoder.joblib',
    test_size=0.15,
    val_size=0.15,
    seed=42,  # Added seed parameter
    use_amp=False  # Added mixed precision flag
):
    # Set seed for reproducibility
    set_seed(seed)

    # Set up logger
    logger = setup_logger(f'{checkpoint_dir}/lstm.log')
    logger.info("Starting LSTM training process...")

    # ------------------ Data Loading & Preprocessing ------------------ #
    logger.info("Loading and processing data with DataHandler...")
    data_handler = DataHandler(
        data_path=data_path,
        feature_columns=feature_columns,
        target_column=target_column,
        traj_id_column=traj_id_column,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,  # Use seed as random_state for reproducibility
        chunksize=10**6,
        seed=seed  # Pass seed to DataHandler
    )

    data_handler.load_and_process_data()
    data_handler.save_preprocessors(scaler_path, label_encoder_path)
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Label encoder saved to {label_encoder_path}")

    dataloaders = data_handler.get_dataloaders(batch_size=batch_size, num_workers=num_workers)
    logger.info("Dataloaders created for train, val, and test sets.")

    # ------------------ Model Setup ------------------ #
    input_size = len(feature_columns)
    num_classes = len(data_handler.label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = LSTMTripClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)
    logger.info("LSTM model initialized.")

    # ------------------ Loss & Optimizer ------------------ #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info("Criterion (CrossEntropyLoss) and Adam optimizer initialized.")

    # ------------------ Training ------------------ #
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        max_grad_norm=max_grad_norm,
        logger=logger,  # <-- pass the logger to Trainer
        use_amp=use_amp  # <-- Enable or disable mixed precision
    )
    logger.info(f"Begin model training for up to {num_epochs} epochs...")
    trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=num_epochs)
    logger.info("Training completed.")

    # ------------------ Test Evaluation ------------------ #
    logger.info("Evaluating on test set...")
    trainer.evaluate(dataloaders['test'], data_handler.label_encoder)
    logger.info("Evaluation completed. LSTM script finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM model for trip classification.")

    # Required
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")

    # Optional
    parser.add_argument("--checkpoint_dir", type=str, default=".", help="Directory to save model checkpoints.")
    parser.add_argument("--scaler_path", type=str, default='scaler.joblib', help="Path to save the scaler.")
    parser.add_argument("--label_encoder_path", type=str, default='label_encoder.joblib', help="Path to save the label encoder.")
    parser.add_argument("--feature_columns", nargs="+", default=["speed"], help="Feature columns to use.")
    parser.add_argument("--target_column", type=str, default="label", help="Target column.")
    parser.add_argument("--traj_id_column", type=str, default="traj_id", help="Trajectory ID column.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the LSTM.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Max gradient norm for clipping.")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set size as a fraction.")
    parser.add_argument("--val_size", type=float, default=0.15, help="Validation set size as a fraction.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_amp", action='store_true', help="Enable mixed precision training.")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
        traj_id_column=args.traj_id_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        scaler_path=args.scaler_path,
        label_encoder_path=args.label_encoder_path,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.random_state,
        use_amp=args.use_amp  # <-- Pass the mixed precision flag
    )
