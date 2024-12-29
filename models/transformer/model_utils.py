# model_utils.py

import torch
import torch.nn as nn
from tqdm import tqdm

class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        feature_size,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        window_size=100,
        dropout=0.1
    ):
        super().__init__()
        self.embedding = nn.Linear(feature_size, d_model)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.position_embedding = nn.Embedding(window_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            activation='relu',
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention-based pooling
        self.attention_weights_layer = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, position_ids, src_key_padding_mask=None):
        """
        x shape: [batch_size, window_size, feature_dim]
        position_ids shape: [batch_size, window_size]
        src_key_padding_mask shape: [batch_size, window_size] -> True for PAD
        """
        # (1) Embedding
        x = self.embedding(x)
        x = self.activation(x)
        x = self.layer_norm(x)

        # (2) Add position embeddings
        pos_emb = self.position_embedding(position_ids)
        x = x + pos_emb

        # (3) Prepare for transformer encoder
        x = x.transpose(0, 1)  # => [window_size, batch_size, d_model]

        # (4) Pass through the Transformer
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # => [batch_size, window_size, d_model]

        # (5) Attention-based pooling
        attention_scores = self.attention_weights_layer(output).squeeze(-1)  # => [batch_size, window_size]
        if src_key_padding_mask is not None:
            # Where mask == True (PAD), set to -inf
            mask = src_key_padding_mask.bool()
            attention_scores[mask] = float('-inf')

        attention_weights = torch.softmax(attention_scores, dim=-1)
        pooled = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)  # => [batch_size, d_model]

        # (6) Classify
        logits = self.classifier(pooled)  # => [batch_size, num_classes]
        return logits


class TrajectoryModel:
    """
    Encapsulates the model, loss, optimizer, device, plus methods for train/evaluate/test/predict.
    """
    def __init__(self, feature_columns, label_encoder, device=None):
        self.feature_columns = feature_columns
        self.label_encoder = label_encoder
        self.num_classes = len(label_encoder.classes_)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def prepare_model(
        self,
        window_size=100,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        lr=1e-4,
        weight_decay=1e-4
    ):
        feature_size = len(self.feature_columns)
        self.model = TrajectoryTransformer(
            feature_size=feature_size,
            num_classes=self.num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            window_size=window_size,
            dropout=dropout
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_one_epoch(self, train_loader, gradient_clip=1.0):
        self.model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sequences, masks, labels in tqdm(train_loader, desc='Training'):
            sequences = sequences.to(self.device)
            masks = masks.to(self.device) if masks is not None else None
            labels = labels.to(self.device)

            position_ids = torch.arange(sequences.size(1), device=self.device).unsqueeze(0).repeat(sequences.size(0), 1)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(sequences, position_ids, src_key_padding_mask=masks)
                loss = self.criterion(outputs, labels)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()

        avg_loss = running_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for sequences, masks, labels in data_loader:
                sequences = sequences.to(self.device)
                masks = masks.to(self.device) if masks is not None else None
                labels = labels.to(self.device)

                position_ids = torch.arange(sequences.size(1), device=self.device).unsqueeze(0).repeat(sequences.size(0), 1)

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(sequences, position_ids, src_key_padding_mask=masks)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def predict(self, data_loader):
        self.model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sequences, masks, labels in data_loader:
                sequences = sequences.to(self.device)
                masks = masks.to(self.device) if masks is not None else None
                labels = labels.to(self.device)

                position_ids = torch.arange(sequences.size(1), device=self.device).unsqueeze(0).repeat(sequences.size(0), 1)

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(sequences, position_ids, src_key_padding_mask=masks)
                    _, predicted = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        return all_labels, all_preds

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
