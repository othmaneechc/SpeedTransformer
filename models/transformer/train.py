# train.py

import argparse
import sys
import logging
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from data_utils import DataProcessor, TripDataset
from model_utils import TrajectoryModel


def setup_logger(log_file='train.log'):
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


def set_seed(seed):
    """
    Set the seed for all relevant libraries to ensure reproducibility.
    """
    import os
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CPU
    torch.use_deterministic_algorithms(True)

    # For CUDA (optional, can make some operations slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables (must be set before any CUDA operations)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():

    import os

    # Set PYTHONHASHSEED for hash-based operations
    os.environ['PYTHONHASHSEED'] = '0'

    # Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    
    parser = argparse.ArgumentParser(description="Train a Trajectory Transformer model.")
    
    # Data
    parser.add_argument("--checkpoint_dir", type=str, default=".", help="Directory to save model checkpoints.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--feature_columns', nargs='+', default=['speed'], help='List of feature columns to use.')
    parser.add_argument('--target_column', type=str, default='label', help='Name of the target column.')
    parser.add_argument('--traj_id_column', type=str, default='traj_id', help='Name of the trajectory ID column.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion of data for testing.')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion of data for validation.')
    parser.add_argument('--random_state', type=int, default=316, help='Random seed.')
    parser.add_argument('--chunksize', type=int, default=10**6, help='Chunksize for reading CSV in chunks.')
    parser.add_argument('--window_size', type=int, default=200, help='Sliding window size.')
    parser.add_argument('--stride', type=int, default=50, help='Stride for the sliding window.')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Maximum number of training epochs.')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience.')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=128, help='Transformer embedding dimension.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer encoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the transformer.')
    
    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate for AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value.')

    # Saving paths
    parser.add_argument('--save_model_path', type=str, default='best_model.pth', help='Where to save the best model weights.')
    parser.add_argument('--save_label_encoder_path', type=str, default='label_encoder.joblib', help='Where to save the fitted label encoder.')
    parser.add_argument('--save_scaler_path', type=str, default='scaler.joblib', help='Where to save the fitted scaler.')

    args = parser.parse_args()

    logger = setup_logger(f'{args.checkpoint_dir}/train.log')

    # **Set Seeds for Reproducibility**
    logger.info(f"Setting random seed to {args.random_state}")
    set_seed(args.random_state)

    # 1) Data Processing
    logger.info("Initializing DataProcessor...")
    processor = DataProcessor(
        data_path=args.data_path,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
        traj_id_column=args.traj_id_column,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        chunksize=args.chunksize,
        window_size=args.window_size,
        stride=args.stride
    )
    processor.load_and_process_data()

    # Save the fitted scaler & label encoder
    processor.save_preprocessors(args.save_scaler_path, args.save_label_encoder_path)
    logger.info(f"Scaler saved to {args.save_scaler_path}")
    logger.info(f"Label encoder saved to {args.save_label_encoder_path}")

    # Build Datasets
    train_dataset = TripDataset(processor.train_sequences, processor.train_labels, processor.train_masks)
    val_dataset   = TripDataset(processor.val_sequences,   processor.val_labels,   processor.val_masks)
    test_dataset  = TripDataset(processor.test_sequences,  processor.test_labels,  processor.test_masks)

    # Set up a generator with the fixed seed for DataLoader
    g = torch.Generator()
    g.manual_seed(args.random_state)
    
    # Build DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(args.random_state + worker_id),
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,   
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(args.random_state + worker_id)
    )
    test_loader = DataLoader(
        test_dataset,  
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(args.random_state + worker_id)
    )

    logger.info("Datasets and DataLoaders ready.")

    # 2) Model Initialization
    logger.info("Initializing TrajectoryModel...")
    model = TrajectoryModel(
        feature_columns=args.feature_columns,
        label_encoder=processor.label_encoder,
        use_amp=False
    )
    model.prepare_model(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    logger.info("Model initialization completed.")

    # 3) Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0

    logger.info("Starting training loop...")
    for epoch in range(args.num_epochs):
        train_loss, train_acc = model.train_one_epoch(train_loader, gradient_clip=args.gradient_clip)
        val_loss, val_acc = model.evaluate(val_loader)

        logger.info(
            f"[Epoch {epoch+1}/{args.num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            model.save_model(args.save_model_path)
            logger.info("  -> Best model saved")
        else:
            epochs_no_improve += 1
            logger.info(f"  -> No improvement ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered.")
            break

    # 4) Test Evaluation
    logger.info("\nLoading best model and evaluating on test set...")
    model.load_model(args.save_model_path)
    test_loss, test_acc = model.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Classification Report & Confusion Matrix
    all_labels, all_preds = model.predict(test_loader)
    report = classification_report(all_labels, all_preds, target_names=processor.label_encoder.classes_)
    logger.info("Classification Report:\n" + report)

    cm = confusion_matrix(all_labels, all_preds)
    logger.info("Saving confusion matrix as confusion_matrix.png...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=processor.label_encoder.classes_,
                yticklabels=processor.label_encoder.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    # plt.show()

    from sklearn.metrics import accuracy_score, recall_score
    exact_accuracy = accuracy_score(all_labels, all_preds)
    exact_recall = recall_score(all_labels, all_preds, average='macro')
    logger.info(f"Exact Test Accuracy: {exact_accuracy:.4f}, Exact Test Recall: {exact_recall:.4f}")

    logger.info("Training script completed successfully.")


if __name__ == "__main__":
    main()
