# random_forest_train.py

import argparse
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_utils import DataProcessor


def setup_logger(log_file='train_random_forest.log'):
    """
    Set up a logger to write logs to a file and to the console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if the logger already exists
    if not logger.handlers:
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
    logger = setup_logger('train_random_forest.log')

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a Random Forest model for trajectory prediction.")

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file for training.')
    parser.add_argument('--feature_columns', nargs='+', default=['speed'], help='List of feature columns to use.')
    parser.add_argument('--target_column', type=str, default='label', help='Name of the target column.')
    parser.add_argument('--traj_id_column', type=str, default='traj_id', help='Name of the trajectory ID column.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Proportion of data for testing.')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion of data for validation.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed.')
    parser.add_argument('--chunksize', type=int, default=10**6, help='Chunksize for reading CSV in chunks.')
    parser.add_argument('--window_size', type=int, default=200, help='Sliding window size.')
    parser.add_argument('--stride', type=int, default=50, help='Stride for the sliding window.')

    # Training arguments
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the Random Forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the trees.')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum number of samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum number of samples required to be at a leaf node.')
    parser.add_argument('--bootstrap', type=bool, default=True, help='Whether bootstrap samples are used when building trees.')
    parser.add_argument('--class_weight', type=str, default='balanced', help='Weights associated with classes.')

    # Saving paths
    parser.add_argument('--save_model_path', type=str, default='random_forest_model.joblib',
                        help='Path to save the trained Random Forest model.')
    parser.add_argument('--save_label_encoder_path', type=str, default='label_encoder.joblib',
                        help='Path to save the fitted label encoder.')
    parser.add_argument('--save_scaler_path', type=str, default='scaler.joblib',
                        help='Path to save the fitted scaler.')

    args = parser.parse_args()

    logger.info("Starting Random Forest training pipeline.")

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

    # Retrieve data splits
    X_train, y_train, X_val, y_val, X_test, y_test = processor.get_data_splits()
    logger.info(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2) Model Initialization
    logger.info("Initializing Random Forest classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        bootstrap=args.bootstrap,
        class_weight=args.class_weight,
        random_state=args.random_state,
        n_jobs=-1  # Utilize all available cores
    )
    logger.info(f"Random Forest parameters: {rf_classifier.get_params()}")

    # 3) Training
    logger.info("Starting model training...")
    rf_classifier.fit(X_train, y_train)
    logger.info("Model training completed.")

    # 4) Validation
    logger.info("Evaluating model on validation set...")
    val_predictions = rf_classifier.predict(X_val)
    val_accuracy = np.mean(val_predictions == y_val)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    val_report = classification_report(y_val, val_predictions, target_names=processor.label_encoder.classes_)
    logger.info(f"Validation Classification Report:\n{val_report}")

    # 5) Testing
    logger.info("Evaluating model on test set...")
    test_predictions = rf_classifier.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    test_report = classification_report(y_test, test_predictions, target_names=processor.label_encoder.classes_)
    logger.info(f"Test Classification Report:\n{test_report}")

    # Confusion Matrix
    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=processor.label_encoder.classes_,
                yticklabels=processor.label_encoder.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Random Forest)')
    plt.savefig('confusion_matrix_random_forest.png')
    logger.info("Confusion matrix saved as 'confusion_matrix_random_forest.png'.")

    # 6) Saving the Model
    logger.info(f"Saving the trained Random Forest model to {args.save_model_path}...")
    joblib.dump(rf_classifier, args.save_model_path)
    logger.info("Model saved successfully.")

    logger.info("Random Forest training pipeline completed successfully.")


if __name__ == "__main__":
    main()
