# fine_tune.py

import argparse
import sys
import logging
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm  # optional

from data_utils import DataProcessor, TripDataset
from model_utils import TrajectoryModel


def setup_logger(log_file='fine_tune.log'):
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
    # Initialize logger
    logger = setup_logger('fine_tune.log')

    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained Trajectory Transformer model.")

    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV data file for fine-tuning.')
    parser.add_argument('--feature_columns', nargs='+', default=['speed'],
                        help='List of feature columns to use.')
    parser.add_argument('--target_column', type=str, default='label',
                        help='Name of the target column.')
    parser.add_argument('--traj_id_column', type=str, default='traj_id',
                        help='Name of the trajectory ID column.')
    parser.add_argument('--test_size', type=float, default=0.7,
                        help='Proportion of data for testing.')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Proportion of data for validation.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--chunksize', type=int, default=10**6,
                        help='Chunksize for reading CSV in chunks.')
    parser.add_argument('--window_size', type=int, default=200,
                        help='Sliding window size.')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride for the sliding window.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Max number of fine-tuning epochs.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience.')

    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer embedding dimension.')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer encoder layers.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate in the transformer.')
    
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='Learning rate for fine-tuning.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for fine-tuning.')
    parser.add_argument('--gradient_clip', type=float, default=0.5,
                        help='Gradient clipping value.')

    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help='Path to the saved pretrained model to fine-tune.')
    parser.add_argument('--save_model_path', type=str, default='fine_tuned_model.pth',
                        help='Where to save the fine-tuned model weights.')
    parser.add_argument('--label_encoder_path', type=str, default='label_encoder.joblib',
                        help='Path to the saved label encoder from previous training.')

    args = parser.parse_args()

    # -----------------------------
    # 1) Data Processing
    # -----------------------------
    logger.info("Initializing DataProcessor for fine-tuning...")
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
    logger.info("Data loaded and processed successfully.")

    logger.info(f"Loading label encoder from {args.label_encoder_path}...")
    loaded_encoder = joblib.load(args.label_encoder_path)
    processor.label_encoder = loaded_encoder
    logger.info("Label encoder loaded and assigned to DataProcessor.")

    # Clear old sequences and re-create them using the loaded encoder
    logger.info("Clearing old sequences and re-creating them using loaded encoder...")
    processor.train_sequences.clear()
    processor.train_labels.clear()
    processor.train_masks.clear()
    processor.val_sequences.clear()
    processor.val_labels.clear()
    processor.val_masks.clear()
    processor.test_sequences.clear()
    processor.test_labels.clear()
    processor.test_masks.clear()
    processor.create_sequences()
    logger.info("Sequences re-created successfully.")

    # Build datasets
    train_dataset = TripDataset(processor.train_sequences, processor.train_labels, processor.train_masks)
    val_dataset   = TripDataset(processor.val_sequences,   processor.val_labels,   processor.val_masks)
    test_dataset  = TripDataset(processor.test_sequences,  processor.test_labels,  processor.test_masks)

    # Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info("DataLoaders for train, val, and test are ready.")

    # -----------------------------
    # 2) Load Pretrained Model
    # -----------------------------
    logger.info("Initializing model for fine-tuning...")
    model = TrajectoryModel(
        feature_columns=args.feature_columns,
        label_encoder=processor.label_encoder
    )
    
    model.prepare_model(
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    logger.info(f"Loading pretrained model from: {args.pretrained_model_path}")
    model.load_model(args.pretrained_model_path)
    logger.info("Pretrained model loaded successfully.")

    # (Optional) Freeze certain layers if desired
    for name, param in model.model.named_parameters():
        if "transformer_encoder.layers.0" in name or "transformer_encoder.layers.1" in name:
            param.requires_grad = False
    logger.info("Optional: Frozen first two layers of the transformer encoder.")

    # Re-initialize optimizer to only update unfrozen params
    model.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    logger.info("Optimizer re-initialized to only update unfrozen parameters.")

    # -----------------------------
    # 3) Fine-Tuning
    # -----------------------------
    logger.info("Starting the fine-tuning process...")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        train_loss, train_acc = model.train_one_epoch(train_loader, gradient_clip=args.gradient_clip)
        val_loss, val_acc = model.evaluate(val_loader)

        logger.info(
            f"[Epoch {epoch+1}/{args.num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Monitor validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            model.save_model(args.save_model_path)
            logger.info("  -> Best fine-tuned model saved.")
        else:
            epochs_no_improve += 1
            logger.info(f"  -> No improvement ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered.")
            break

    # -----------------------------
    # 4) Final Test
    # -----------------------------
    logger.info(f"\nLoading best fine-tuned model from {args.save_model_path} for final evaluation...")
    model.load_model(args.save_model_path)
    test_loss, test_acc = model.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    all_labels, all_preds = model.predict(test_loader)
    report = classification_report(all_labels, all_preds, target_names=processor.label_encoder.classes_)
    logger.info("Classification Report:\n" + report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=processor.label_encoder.classes_,
                yticklabels=processor.label_encoder.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Fine-Tuned Model)')
    plt.savefig('confusion_matrix_fine_tune.png')
    logger.info("Confusion matrix saved as 'confusion_matrix_fine_tune.png'.")

    logger.info("Fine-tuning script completed successfully.")


if __name__ == "__main__":
    main()
