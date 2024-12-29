# fine_tune.py

import os
import argparse
import torch
import torch.nn as nn
import joblib
from data_utils import DataHandler
from models import LSTMTripClassifier
from trainer import Trainer

# ------------------ FineTuner Class ------------------ #
class FineTuner:
    def __init__(self, model, fine_tune_layers=None, freeze_layers=True):
        """
        Initializes the FineTuner.

        Args:
            model (nn.Module): The pre-trained PyTorch model.
            fine_tune_layers (list, optional): List of layer name substrings to fine-tune.
                                                If None, all layers are fine-tuned.
            freeze_layers (bool, optional): Whether to freeze layers not in fine_tune_layers.
                                            Defaults to True.
        """
        self.model = model
        self.fine_tune_layers = fine_tune_layers
        self.freeze_layers = freeze_layers
        self.freeze_model()

    def freeze_model(self):
        """
        Freezes the layers of the model that are not in fine_tune_layers.
        If fine_tune_layers is None, all layers are trainable.
        """
        if self.fine_tune_layers is None:
            print("Fine-tuning all layers.")
            for param in self.model.parameters():
                param.requires_grad = True
            return  # All layers are trainable

        print(f"Fine-tuning layers containing: {self.fine_tune_layers}")
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in self.fine_tune_layers):
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")
            else:
                param.requires_grad = False
                print(f"Frozen layer: {name}")

# ------------------ Main Function ------------------ #
def main(
    pre_trained_model_path,
    fine_tune_data_path,
    scaler_path,
    label_encoder_path,
    feature_columns,
    target_column,
    traj_id_column='traj_id',
    fine_tune_layers=None,
    freeze_layers=True,
    batch_size=128,
    num_workers=4,
    learning_rate=1e-4,
    weight_decay=1e-5,
    num_epochs=20,
    patience=5,
    max_grad_norm=5.0,
    checkpoint_dir='fine_tuned_checkpoints',
    test_size=0.15,
    val_size=0.10,
):
    """
    Fine-tunes an LSTMTripClassifier using a pre-trained model.

    Args:
        pre_trained_model_path (str): Path to the pre-trained model checkpoint.
        fine_tune_data_path (str): Path to the CSV data file for fine-tuning.
        scaler_path (str): Path to the saved scaler joblib file.
        label_encoder_path (str): Path to the saved label encoder joblib file.
        feature_columns (list): List of feature column names.
        target_column (str): Name of the target column.
        traj_id_column (str): Name of the trajectory ID column.
        fine_tune_layers (list): Substrings in layer names to fine-tune. If None, tune all.
        freeze_layers (bool): Whether to freeze layers not listed in fine_tune_layers.
        batch_size (int): Batch size for fine-tuning.
        num_workers (int): Number of workers for DataLoader.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 penalty).
        num_epochs (int): Max number of epochs for fine-tuning.
        patience (int): Early-stopping patience.
        max_grad_norm (float): Gradient clipping norm.
        checkpoint_dir (str): Directory to save fine-tuned model checkpoints.
        test_size (float): Fraction of dataset for testing.
        val_size (float): Fraction of dataset for validation.
    """
    print("Loading scaler and label encoder...")
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    print("Scaler and Label Encoder loaded successfully.")

    print("Initializing DataHandler for fine-tuning data...")
    data_handler = DataHandler(
        data_path=fine_tune_data_path,
        feature_columns=feature_columns,
        target_column=target_column,
        traj_id_column=traj_id_column,
        test_size=test_size,
        val_size=val_size,
        random_state=42,
        chunksize=10**6,
    )
    data_handler.scaler = scaler
    data_handler.label_encoder = label_encoder

    print("Loading and processing fine-tuning data...")
    data_handler.load_and_process_data()

    print("Creating DataLoaders...")
    dataloaders = data_handler.get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    input_size = len(feature_columns)
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Must match original model's architecture
    model = LSTMTripClassifier(
        input_size=input_size,
        hidden_size=256,  # Must match pre-trained
        num_layers=2,     # Must match pre-trained
        num_classes=num_classes,
        dropout=0.3       # Must match pre-trained
    ).to(device)

    print(f"Loading pre-trained model from '{pre_trained_model_path}'...")
    try:
        model.load_state_dict(torch.load(pre_trained_model_path, map_location=device))
        print("Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return

    print("Initializing FineTuner...")
    fine_tuner = FineTuner(model, fine_tune_layers=fine_tune_layers, freeze_layers=freeze_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fine_tuner.model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print("Optimizer initialized.")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        max_grad_norm=max_grad_norm,
    )

    print("Starting fine-tuning...")
    trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=num_epochs)

    print("\nEvaluating on Test Set...")
    trainer.evaluate(dataloaders['test'], label_encoder)

# ------------------ Execution Block ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LSTM model.")

    parser.add_argument('--pre_trained_model_path', type=str, required=True,
                        help='Path to the pre-trained LSTM model checkpoint (e.g. best_model.pth).')
    parser.add_argument('--fine_tune_data_path', type=str, required=True,
                        help='Path to the CSV data file for fine-tuning.')
    parser.add_argument('--scaler_path', type=str, required=True,
                        help='Path to the saved scaler (e.g., scaler.joblib).')
    parser.add_argument('--label_encoder_path', type=str, required=True,
                        help='Path to the saved label encoder (e.g., label_encoder.joblib).')
    parser.add_argument('--feature_columns', nargs='+', default=['speed'],
                        help='List of feature columns in the dataset.')
    parser.add_argument('--target_column', type=str, default='label',
                        help='Name of the target column.')
    parser.add_argument('--traj_id_column', type=str, default='traj_id',
                        help='Name of the trajectory ID column.')
    parser.add_argument('--fine_tune_layers', nargs='+', default=None,
                        help='List of layer-name substrings to fine-tune. If None, tune all.')
    parser.add_argument('--freeze_layers', action='store_true', default=False,
                        help='Freeze layers not in fine_tune_layers. Default False means all are trainable.')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization).')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of fine-tuning epochs.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='Gradient clipping norm.')
    parser.add_argument('--checkpoint_dir', type=str, default='fine_tuned_checkpoints',
                        help='Directory to save fine-tuned model checkpoints.')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Proportion of the data to use for test.')
    parser.add_argument('--val_size', type=float, default=0.10,
                        help='Proportion of the data to use for validation.')

    args = parser.parse_args()

    main(
        pre_trained_model_path=args.pre_trained_model_path,
        fine_tune_data_path=args.fine_tune_data_path,
        scaler_path=args.scaler_path,
        label_encoder_path=args.label_encoder_path,
        feature_columns=args.feature_columns,
        target_column=args.target_column,
        traj_id_column=args.traj_id_column,
        fine_tune_layers=args.fine_tune_layers,
        freeze_layers=args.freeze_layers,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        test_size=args.test_size,
        val_size=args.val_size,
    )
