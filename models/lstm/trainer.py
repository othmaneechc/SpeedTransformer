import os
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        checkpoint_dir='model_checkpoints',
        patience=7,
        max_grad_norm=5.0,
        logger=None,  # We'll use the logger argument
        use_amp=False
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        self.logger = logger or logging.getLogger(__name__)

        self.best_val_accuracy = 0.0
        self.trigger_times = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision (AMP) is enabled.")
        else:
            self.scaler = None

    def train_epoch(self, dataloader):
        self.model.train()
        cumulative_loss = 0.0
        correct = 0
        total_samples = 0

        for sequences, labels, lengths, masks in tqdm(dataloader, desc="Training", total=len(dataloader)):
            batch_size = labels.size(0)

            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            masks = masks.to(self.device)

            optimizer = self.optimizer

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, _ = self.model(sequences, lengths, mask=masks)
                    loss = self.criterion(outputs, labels)
            else:
                outputs, _ = self.model(sequences, lengths, mask=masks)
                loss = self.criterion(outputs, labels)

            optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

            cumulative_loss += loss.item() * batch_size
            total_samples += batch_size

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

        avg_loss = (cumulative_loss / total_samples) if total_samples else 0.0
        accuracy = (100.0 * correct / total_samples) if total_samples else 0.0
        return avg_loss, accuracy

    def validate_epoch(self, dataloader):
        self.model.eval()
        cumulative_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for sequences, labels, lengths, masks in tqdm(dataloader, desc="Validation", total=len(dataloader)):
                batch_size = labels.size(0)

                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                masks = masks.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(sequences, lengths, mask=masks)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs, _ = self.model(sequences, lengths, mask=masks)
                    loss = self.criterion(outputs, labels)

                cumulative_loss += loss.item() * batch_size
                total_samples += batch_size

                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()

        avg_loss = (cumulative_loss / total_samples) if total_samples else 0.0
        accuracy = (100.0 * correct / total_samples) if total_samples else 0.0
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs):
        """
        Full training loop, including early stopping based on best validation accuracy.
        """
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            val_loss, val_acc = self.validate_epoch(val_loader)
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Early stopping logic
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.trigger_times = 0
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Best model saved with validation accuracy: {self.best_val_accuracy:.2f}%")
            else:
                self.trigger_times += 1
                self.logger.info(f"No improvement for {self.trigger_times} epoch(s).")
                if self.patience is not None and self.trigger_times >= self.patience:
                    self.logger.info("Early stopping triggered!")
                    break

    def evaluate(self, dataloader, label_encoder):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels, lengths, masks in tqdm(dataloader, desc="Testing", total=len(dataloader)):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)
                masks = masks.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(sequences, lengths, mask=masks)
                else:
                    outputs, _ = self.model(sequences, lengths, mask=masks)

                predicted = outputs.argmax(dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        self.logger.info(f"Test Accuracy: {test_accuracy:.2f}%")

        # Classification report
        report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
        self.logger.info("Classification Report:\n" + report)

        # Confusion matrix
        conf_mat = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_,
                    cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
