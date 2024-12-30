# random_forest_finetune.py

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


def setup_logger(log_file='finetune_random_forest.log'):
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
    logger = setup_logger('finetune_random_forest.log')

    # Argument parsing
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained Random Forest model for trajectory prediction.")

    # Data arguments
    parser.add_argument('--new_data_path', type=str, required=True,
                        help='Path to the new CSV data file for fine-tuning.')
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
                        help='Random seed for reproducibility.')
    parser.add_argument('--chunksize', type=int, default=10**6,
                        help='Chunksize for reading CSV in chunks.')
    parser.add_argument('--window_size', type=int, default=200,
                        help='Sliding window size.')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride for the sliding window.')

    # Model arguments
    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help='Path to the pre-trained Random Forest model file (joblib).')
    parser.add_argument('--pretrained_scaler_path', type=str, required=True,
                        help='Path to the pre-trained scaler (joblib).')
    parser.add_argument('--pretrained_label_encoder_path', type=str, required=True,
                        help='Path to the pre-trained label encoder (joblib).')

    # Fine-tuning parameters
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of trees in the Random Forest.')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum depth of the trees.')
    parser.add_argument('--min_samples_split', type=int, default=2,
                        help='Minimum number of samples required to split an internal node.')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                        help='Minimum number of samples required to be at a leaf node.')
    parser.add_argument('--bootstrap', type=bool, default=True,
                        help='Whether bootstrap samples are used when building trees.')
    parser.add_argument('--class_weight', type=str, default='balanced',
                        help='Weights associated with classes.')

    # Saving paths
    parser.add_argument('--save_model_path', type=str, default='random_forest_finetuned_model.joblib',
                        help='Path to save the fine-tuned Random Forest model.')
    parser.add_argument('--save_label_encoder_path', type=str, default='label_encoder_finetuned.joblib',
                        help='Path to save the fitted label encoder (if updated).')
    parser.add_argument('--save_scaler_path', type=str, default='scaler_finetuned.joblib',
                        help='Path to save the fitted scaler (if updated).')

    args = parser.parse_args()

    logger.info("Starting Random Forest fine-tuning pipeline.")

    # 1) Load Pre-trained Model and Preprocessors
    logger.info(f"Loading pre-trained Random Forest model from {args.pretrained_model_path}...")
    pretrained_rf = joblib.load(args.pretrained_model_path)
    logger.info("Pre-trained model loaded successfully.")

    logger.info(f"Loading pre-trained scaler from {args.pretrained_scaler_path}...")
    scaler = joblib.load(args.pretrained_scaler_path)
    logger.info("Pre-trained scaler loaded successfully.")

    logger.info(f"Loading pre-trained label encoder from {args.pretrained_label_encoder_path}...")
    label_encoder = joblib.load(args.pretrained_label_encoder_path)
    logger.info("Pre-trained label encoder loaded successfully.")

    # 2) Data Processing
    logger.info("Initializing DataProcessor for fine-tuning...")
    processor = DataProcessor(
        data_path=args.new_data_path,
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
    processor.scaler = scaler  # Use the pre-trained scaler
    processor.label_encoder = label_encoder  # Use the pre-trained label encoder

    logger.info("Processing new data for fine-tuning...")
    processor.load_and_process_data()
    logger.info("New data loaded and processed successfully.")

    # Retrieve data splits
    X_train_new, y_train_new, X_val_new, y_val_new, X_test_new, y_test_new = processor.get_data_splits()
    logger.info(f"New Training samples: {X_train_new.shape[0]}, Validation samples: {X_val_new.shape[0]}, Test samples: {X_test_new.shape[0]}")

    # 3) Fine-Tuning (Retraining)
    # Note: Random Forest does not support incremental learning. To fine-tune, you typically retrain the model.
    # If you have access to the original training data, consider combining it with the new data for retraining.

    # 4) Initialize a new Random Forest Classifier with desired hyperparameters
    logger.info("Initializing a new Random Forest classifier for fine-tuning...")
    finetuned_rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        bootstrap=args.bootstrap,
        class_weight=args.class_weight,
        random_state=args.random_state,
        n_jobs=-1  # Utilize all available cores
    )
    logger.info("Random Forest classifier initialized.")

    # 5) Fine-tuning (Retraining)
    logger.info("Starting model fine-tuning (retraining) with new data...")
    finetuned_rf.fit(X_train_new, y_train_new)
    logger.info("Model fine-tuning completed.")

    # 6) Evaluation on Validation Set
    logger.info("Evaluating fine-tuned model on validation set...")
    val_predictions = finetuned_rf.predict(X_val_new)
    val_accuracy = np.mean(val_predictions == y_val_new)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    val_report = classification_report(y_val_new, val_predictions, target_names=label_encoder.classes_)
    logger.info(f"Validation Classification Report:\n{val_report}")

    # 7) Evaluation on Test Set
    logger.info("Evaluating fine-tuned model on test set...")
    test_predictions = finetuned_rf.predict(X_test_new)
    test_accuracy = np.mean(test_predictions == y_test_new)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    test_report = classification_report(y_test_new, test_predictions, target_names=label_encoder.classes_)
    logger.info(f"Test Classification Report:\n{test_report}")

    # 8) Confusion Matrix
    logger.info("Generating confusion matrix for test set...")
    cm = confusion_matrix(y_test_new, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Fine-Tuned Random Forest)')
    plt.savefig('confusion_matrix_finetuned_random_forest.png')
    logger.info("Confusion matrix saved as 'confusion_matrix_finetuned_random_forest.png'.")

    # 9) Saving the Fine-Tuned Model and Preprocessors
    logger.info(f"Saving the fine-tuned Random Forest model to {args.save_model_path}...")
    joblib.dump(finetuned_rf, args.save_model_path)
    logger.info("Fine-tuned model saved successfully.")

    # If the label encoder or scaler were updated during processing, save them as well.
    # In this script, we are reusing the existing scaler and label encoder, so no need to save them again.
    # However, if any updates are made, uncomment the following lines:

    # logger.info(f"Saving the updated label encoder to {args.save_label_encoder_path}...")
    # joblib.dump(processor.label_encoder, args.save_label_encoder_path)
    # logger.info("Label encoder saved successfully.")

    # logger.info(f"Saving the updated scaler to {args.save_scaler_path}...")
    # joblib.dump(processor.scaler, args.save_scaler_path)
    # logger.info("Scaler saved successfully.")

    logger.info("Random Forest fine-tuning pipeline completed successfully.")


if __name__ == "__main__":
    main()
