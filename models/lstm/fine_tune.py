# fine_tune.py

import os
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
            freeze_layers (bool, optional): Whether to freeze layers not in fine_tune_layers. Defaults to True.
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
    fine_tune_layers=None,  # List of layer name substrings to fine-tune
    freeze_layers=True,
    batch_size=128,
    num_workers=4,
    learning_rate=1e-4,  # Often lower for fine-tuning
    weight_decay=1e-5,
    num_epochs=20,
    patience=5,
    max_grad_norm=5.0,
    checkpoint_dir='fine_tuned_checkpoints',
    test_size=0.15,  # Adjusted to ensure proper splits
    val_size=0.10,   # Adjusted to ensure proper splits
):
    """
    Main function to fine-tune the LSTMTripClassifier model.

    Args:
        pre_trained_model_path (str): Path to the pre-trained model checkpoint.
        fine_tune_data_path (str): Path to the CSV data file for fine-tuning.
        scaler_path (str): Path to the saved scaler (e.g., 'scaler.joblib').
        label_encoder_path (str): Path to the saved label encoder (e.g., 'label_encoder.joblib').
        feature_columns (list): List of feature column names.
        target_column (str): Name of the target column.
        traj_id_column (str, optional): Name of the trajectory ID column. Defaults to 'traj_id'.
        fine_tune_layers (list, optional): List of layer name substrings to fine-tune.
                                           If None, fine-tune all layers.
        freeze_layers (bool, optional): Whether to freeze layers not in fine_tune_layers. Defaults to True.
        batch_size (int, optional): Batch size for fine-tuning. Defaults to 128.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        weight_decay (float, optional): Weight decay (L2 penalty) for optimizer. Defaults to 1e-5.
        num_epochs (int, optional): Maximum number of epochs for fine-tuning. Defaults to 20.
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 5.
        max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 5.0.
        checkpoint_dir (str, optional): Directory to save fine-tuned model checkpoints. Defaults to 'fine_tuned_checkpoints'.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.15.
        val_size (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.10.
    """

    # ------------------ Load Preprocessing Objects ------------------ #
    print("Loading scaler and label encoder...")
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    print("Scaler and Label Encoder loaded successfully.")

    # ------------------ Initialize DataHandler ------------------ #
    print("Initializing DataHandler for fine-tuning data...")
    data_handler = DataHandler(
        data_path=fine_tune_data_path,
        feature_columns=feature_columns,
        target_column=target_column,
        traj_id_column=traj_id_column,
        test_size=test_size,
        val_size=val_size,
        random_state=42,
        chunksize=10**6,  # Adjust based on memory constraints
    )
    data_handler.scaler = scaler  # Use pre-loaded scaler
    data_handler.label_encoder = label_encoder  # Use pre-loaded label encoder

    # ------------------ Load and Process Data ------------------ #
    print("Loading and processing fine-tuning data...")
    data_handler.load_and_process_data()

    # ------------------ Create DataLoaders ------------------ #
    print("Creating DataLoaders for fine-tuning...")
    dataloaders = data_handler.get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # ------------------ Initialize Model ------------------ #
    input_size = len(feature_columns)
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # It's crucial that the model architecture matches the pre-trained model
    # Ensure that hidden_size, num_layers, etc., are consistent
    # If unsure, consider loading the model without re-initializing
    # Here, we assume the architecture is known and consistent

    # Load the model architecture
    model = LSTMTripClassifier(
        input_size=input_size,
        hidden_size=256,      # Must match pre-trained model's hidden_size
        num_layers=2,         # Must match pre-trained model's num_layers
        num_classes=num_classes,
        dropout=0.3           # Must match pre-trained model's dropout
    )
    model = model.to(device)

    # ------------------ Load Pre-trained Model Checkpoint ------------------ #
    print(f"Loading pre-trained model checkpoint from '{pre_trained_model_path}'...")
    try:
        model.load_state_dict(torch.load(pre_trained_model_path, map_location=device))
        print("Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        return

    # ------------------ Initialize FineTuner ------------------ #
    print("Initializing FineTuner...")
    fine_tuner = FineTuner(model, fine_tune_layers=fine_tune_layers, freeze_layers=freeze_layers)

    # ------------------ Define Loss and Optimizer ------------------ #
    criterion = nn.CrossEntropyLoss()

    # Only parameters that require gradients are passed to the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fine_tuner.model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print("Optimizer initialized.")

    # ------------------ Initialize Trainer ------------------ #
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        max_grad_norm=max_grad_norm,
    )

    # ------------------ Start Fine-Tuning ------------------ #
    print("Starting fine-tuning process...")
    trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=num_epochs)

    # ------------------ Evaluate Fine-Tuned Model ------------------ #
    print("\nEvaluating Fine-Tuned Model on Test Set:")
    trainer.evaluate(dataloaders['test'], label_encoder)

# ------------------ Execution Block ------------------ #
if __name__ == "__main__":
    # ------------------ Define Parameters Here ------------------ #
    # Specify the paths to your files
    pre_trained_model_path = '/data/A-SUBMISSION/experiments/training/mobis_lstm/best_model.pth'  # Path to your pre-trained model
    fine_tune_data_path = '/data/A-SUBMISSION/data/processed_geolife_updated4.csv'
    scaler_path = '/data/A-SUBMISSION/experiments/training/mobis_lstm/scaler.joblib'  # Path to your saved scaler
    label_encoder_path = '/data/A-SUBMISSION/experiments/training/mobis_lstm/label_encoder.joblib'  # Path to your saved label encoder

    # Define your feature columns and target column
    feature_columns = [
        # 'lat', 'long', 'hour', 'day_of_week', 'time_since_start',
        # 'delta_lat', 'delta_long', 'distance', 'bearing',
        # 'time_diff', 
        'speed', 
        # 'acceleration', 'cumulative_distance',
        # 'speed_roll_mean', 'speed_roll_std', 'acceleration_roll_mean',
        # 'acceleration_roll_std', 'bearing_change', 'remaining_distance',
    ]
    target_column = 'label'
    traj_id_column = 'traj_id'

    # Define which layers to fine-tune (optional)
    # If None, all layers will be fine-tuned
    # To fine-tune specific layers, provide a list of substrings in layer names
    # For example, to fine-tune only the fully connected layer:
    # fine_tune_layers = ['fc']
    fine_tune_layers = None  # Fine-tune all layers

    # Define model parameters (should match the pre-trained model's parameters)
    # It's crucial that these match the pre-trained model to ensure compatibility
    hidden_size = 256
    num_layers = 2
    dropout = 0.3

    # Define DataLoader parameters
    batch_size = 128
    num_workers = 12  # Adjust based on your system's capabilities

    # Define optimizer and training parameters
    learning_rate = 1e-4  # Often lower for fine-tuning
    weight_decay = 1e-5
    num_epochs = 20
    patience = 5
    max_grad_norm = 5.0
    checkpoint_dir = 'fine_tuned_checkpoints'

    # Define data split ratios
    test_size = 0.75  # 15% for testing
    val_size = 0.2   # 10% for validation
    # This leaves 75% for training

    # ------------------ Call the Main Function ------------------ #
    main(
        pre_trained_model_path=pre_trained_model_path,
        fine_tune_data_path=fine_tune_data_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path,
        feature_columns=feature_columns,
        target_column=target_column,
        traj_id_column=traj_id_column,
        fine_tune_layers=fine_tune_layers,
        freeze_layers=True,  # Set to True to freeze layers not in fine_tune_layers
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        max_grad_norm=max_grad_norm,
        checkpoint_dir=checkpoint_dir,
        test_size=test_size,
        val_size=val_size,
    )
