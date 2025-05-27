# test.py

import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from data_utils import DataProcessor, TripDataset
from model_utils import TrajectoryModel


def main():
    parser = argparse.ArgumentParser(description="Test a Trajectory Transformer model.")
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV data file for testing.')
    parser.add_argument('--feature_columns', nargs='+', default=['speed'],
                        help='List of feature columns to use.')
    parser.add_argument('--target_column', type=str, default='label',
                        help='Name of the target column.')
    parser.add_argument('--traj_id_column', type=str, default='traj_id',
                        help='Name of the trajectory ID column.')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Proportion of data for testing.')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Proportion of data for validation.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--chunksize', type=int, default=10**6,
                        help='Chunksize for reading CSV in chunks.')
    parser.add_argument('--window_size', type=int, default=200,
                        help='Sliding window size.')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride for the sliding window.')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')
    
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer embedding dimension.')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer encoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate in the transformer.')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model weights.')
    parser.add_argument('--label_encoder_path', type=str, default='label_encoder.joblib',
                        help='Path to the saved label encoder.')

    args = parser.parse_args()

    # -----------------------------
    # 1) Data Processing
    # -----------------------------
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
    # We'll do the normal pipeline
    processor.load_and_process_data()
    
    # Overwrite the label encoder with the one from disk
    loaded_encoder = joblib.load(args.label_encoder_path)
    processor.label_encoder = loaded_encoder
    print(f"Loaded label encoder from {args.label_encoder_path}")

    # Re-create sequences if needed, because the above call replaced the label encoder
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

    # We only need the test set
    test_dataset = TripDataset(processor.test_sequences, processor.test_labels, processor.test_masks)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # -----------------------------
    # 2) Initialize & Load Model
    # -----------------------------
    model = TrajectoryModel(feature_columns=args.feature_columns,
                            label_encoder=processor.label_encoder)
    model.prepare_model(window_size=args.window_size,
                        d_model=args.d_model,
                        nhead=args.nhead,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
    model.load_model(args.model_path)

    # -----------------------------
    # 3) Evaluate
    # -----------------------------
    test_loss, test_acc = model.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Classification Report and Confusion Matrix
    all_labels, all_preds = model.predict(test_loader)
    report = classification_report(all_labels, all_preds, target_names=processor.label_encoder.classes_)
    print("Classification Report:\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=processor.label_encoder.classes_,
                yticklabels=processor.label_encoder.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Test Script)')
    plt.show()


if __name__ == "__main__":
    main()
