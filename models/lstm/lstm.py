# lstm.py

import os
import torch
import torch.nn as nn
import joblib
import argparse
from data_utils import DataHandler
from models import LSTMTripClassifier
from trainer import Trainer

def main(
    data_path,
    feature_columns,
    target_column,
    traj_id_column='traj_id',
    batch_size=128,
    num_workers=4,
    learning_rate=0.001,
    weight_decay=1e-4,
    hidden_size=256,
    num_layers=2,
    dropout=0.3,
    num_epochs=50,
    patience=7,
    max_grad_norm=5.0,
    checkpoint_dir='last_model',
    scaler_path='last_model/scaler.joblib',
    label_encoder_path='last_model/label_encoder.joblib',
    test_size=0.1,
    val_size=0.2
):
    # ------------------ Data Loading & Preprocessing ------------------ #
    data_handler = DataHandler(
        data_path=data_path,
        feature_columns=feature_columns,
        target_column=target_column,
        traj_id_column=traj_id_column,
        test_size=test_size,
        val_size=val_size,
        random_state=42,
        chunksize=10**6,
    )

    data_handler.load_and_process_data()
    data_handler.save_preprocessors(scaler_path, label_encoder_path)
    dataloaders = data_handler.get_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # ------------------ Model Setup ------------------ #
    input_size = len(feature_columns)
    num_classes = len(data_handler.label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LSTMTripClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    # ------------------ Loss & Optimizer ------------------ #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ------------------ Training ------------------ #
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience,
        max_grad_norm=max_grad_norm,
    )
    trainer.train(dataloaders['train'], dataloaders['val'], num_epochs=num_epochs)

    # ------------------ Test Evaluation ------------------ #
    trainer.evaluate(dataloaders['test'], data_handler.label_encoder)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM model for trip classification.")

    # Required arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")

    # Optional arguments
    parser.add_argument("--checkpoint_dir", type=str, default=".", help="Directory to save model checkpoints.")
    parser.add_argument("--scaler_path", type=str, default='scaler.joblib', help="Path to save the scaler.")
    parser.add_argument("--label_encoder_path", type=str, default='label_encoder.joblib', help="Path to save the label encoder.")
    parser.add_argument("--feature_columns", nargs="+", default=["speed"], help="Feature columns to use.")
    parser.add_argument("--target_column", type=str, default="label", help="Target column.")
    parser.add_argument("--traj_id_column", type=str, default="traj_id", help="Trajectory ID column.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the LSTM.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Max gradient norm for clipping.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size as a fraction.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation set size as a fraction.")

    args = parser.parse_args()

    # Call the main function with parsed arguments
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
    )
