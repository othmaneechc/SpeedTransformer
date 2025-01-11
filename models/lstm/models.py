import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Attention Class ------------------ #
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_outputs, mask=None):
        """
        lstm_outputs: (batch_size, seq_len=200, hidden_size*2)
        mask: (batch_size, 200), 1 for valid, 0 for padded
        """
        # Raw attention scores
        attention_scores = self.attention_layer(lstm_outputs).squeeze(-1)  # (batch_size, seq_len)

        # If mask is provided, set padded positions to -1e9
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-1e9'))

        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        attention_weights = attention_weights.unsqueeze(-1)     # (batch_size, seq_len, 1)

        context_vector = torch.sum(lstm_outputs * attention_weights, dim=1)  # (batch_size, hidden_size*2)
        return context_vector, attention_weights.squeeze(-1)


# ------------------ LSTMTripClassifier Class ------------------ #
class LSTMTripClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(LSTMTripClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = Attention(hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths, mask=None):
        """
        x: (batch_size, 200, input_size)
        lengths: (batch_size,) - actual valid length of each sequence
        mask: (batch_size, 200) - 1 for valid, 0 for padded frames

        Returns:
            out: (batch_size, num_classes)
            attention_weights: (batch_size, 200)
        """

        # If all sequences are zero-length (extreme corner case)
        if lengths.max() == 0:
            batch_size = x.size(0)
            device = x.device
            out = torch.zeros((batch_size, self.fc.out_features), device=device)
            attn_w = torch.zeros((batch_size, x.size(1)), device=device)
            return out, attn_w

        # Use pack_padded_sequence for ignoring padded frames in the LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )
        # lstm_outputs: (batch_size, 200, hidden_size * 2)

        # Apply attention
        context_vector, attention_weights = self.attention(lstm_outputs, mask=mask)

        # Dropout
        context_vector = self.dropout(context_vector)

        # Final linear layer
        out = self.fc(context_vector)
        return out, attention_weights
