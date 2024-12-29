# fine_tuning.py

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import joblib
from tqdm import tqdm

from data_utils import DataHandler
from models import LSTMTripClassifier
from trainer import Trainer


def setup_logger(log_file='fine_tune_lstm.log'):
    """
    Set up a logger to write logs to a file and to the console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if the logger already exists
    if logger.hasHandlers():
        return logger

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    logger = setup_logger('fine_tune_lstm.log')

    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained LSTM model.")

    # Data paths
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the CSV data file for fine-tuning.")
    parser.add_argument("--scaler_path", type=str, default='scaler.joblib',
                        help="Path to the saved scaler from previous training.")
    parser.add_argument("--label_encoder_path", type=str, default='label_encoder.joblib',
                        help="Path to the saved label encoder from previous training.")
    
    # Model hyperparameters
    parser.add_argument("--feature_columns", nargs="+", default=["speed"],
                        help="Feature columns to use.")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden size of the LSTM.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate.")
    
    # Fine-tuning settings
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for fine-tuning.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="Max norm for gradient clipping.")
    
    # Data splits and loader
    parser.add_argument("--target_column", type=str, default="label",
                        help="Target column.")
    parser.add_argument("--traj_id_column", type=str, default="traj_id",
                        help="Trajectory ID column.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers.")
    parser.add_argument("--test_size", type=float, default=0.7,
                        help="Fraction of data for testing.")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Fraction of data for validation.")
    
    # Pretrained model checkpoint
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Path to the saved pretrained LSTM model checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, default="fine_tune_checkpoints",
                        help="Directory to save fine-tuned model checkpoints.")

    args = parser.parse_args()

    # -----------------------------
    # 1) Data Processing
    # -----------------------------
    logger.info("Initializing DataHandler for fine-tuning...")
    data_handler = DataHandler(
        data_path=args.data_path,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
        traj_id_column=args.traj_id_column,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42,     # or any desired seed
        chunksize=10**6,
    )
    
    # Load & process the data (this normally fits a new scaler/encoder).
    # But we'll overwrite the label encoder and scaler with the previously saved ones.
    data_handler.load_and_process_data()
    logger.info("Data loaded and processed successfully.")

    # Overwrite the newly-fitted scaler/encoder with saved ones
    logger.info(f"Loading scaler from {args.scaler_path}...")
    saved_scaler = joblib.load(args.scaler_path)
    data_handler.scaler = saved_scaler
    logger.info("Scaler overwritten with saved scaler.")

    logger.info(f"Loading label encoder from {args.label_encoder_path}...")
    saved_label_encoder = joblib.load(args.label_encoder_path)
    data_handler.label_encoder = saved_label_encoder
    logger.info("Label encoder overwritten with saved encoder.")
    
    # Because we replaced the scaler & encoder, we should re-run the data transformation
    logger.info("Re-building datasets using the loaded scaler/encoder...")
    # Clear the data and re-build sequences (depending on your DataHandler design).
    data_handler.train_sequences.clear()
    data_handler.train_labels.clear()
    data_handler.train_masks.clear()
    data_handler.val_sequences.clear()
    data_handler.val_labels.clear()
    data_handler.val_masks.clear()
    data_handler.test_sequences.clear()
    data_handler.test_labels.clear()
    data_handler.test_masks.clear()
    
    data_handler.create_sequences()  # or whatever method re-applies the transformations
    logger.info("Sequences re-created successfully.")
    
    # Get DataLoaders
    dataloaders = data_handler.get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    logger.info("DataLoaders for train, val, and test are ready.")

    # -----------------------------
    # 2) Load Pretrained Model
    # -----------------------------
    logger.info("Initializing LSTM model for fine-tuning...")
    input_size = len(args.feature_columns)
    num_classes = len(data_handler.label_encoder.classes_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = LSTMTripClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    logger.info(f"Loading pretrained model from {args.pretrained_model_path}...")
    model.load_state_dict(torch.load(args.pretrained_model_path))
    logger.info("Pretrained model weights loaded successfully.")

    # (Optional) Freeze certain LSTM layers if desired, e.g. the first layer:
    # for name, param in model.named_parameters():
    #     if "lstm.weight_ih_l0" in name or "lstm.weight_hh_l0" in name:
    #         param.requires_grad = False
    # logger.info("Optional: Frozen first LSTM layer.")

    # -----------------------------
    # 3) Re-initialize Optimizer / Trainer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    # Filter only params that require grad if any layers are frozen
    params_to_update = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params_to_update, lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info("Optimizer re-initialized for fine-tuning.")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        logger=logger  # Pass the logger to the Trainer
    )

    # -----------------------------
    # 4) Fine-Tuning
    # -----------------------------
    logger.info(f"Starting fine-tuning for up to {args.num_epochs} epochs...")
    trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=args.num_epochs)
    logger.info("Fine-tuning completed.")

    # -----------------------------
    # 5) Final Test
    # -----------------------------
    logger.info("Evaluating the fine-tuned model on the test set...")
    trainer.evaluate(dataloaders['test'], data_handler.label_encoder)
    logger.info("Evaluation completed. Fine-tuning script finished successfully.")


if __name__ == "__main__":
    main()
