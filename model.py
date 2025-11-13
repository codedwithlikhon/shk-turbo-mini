
import os
import json
import torch
from torch.utils.data import Dataset

def build_vocab(data_path):
    """Builds a character vocabulary from the data file."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Cannot build vocab.")
        return 0, {}, {}

    chars = sorted(list(set(text)))
    # Add a padding token
    if '<pad>' not in chars:
        chars.append('<pad>')
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return len(chars), char_to_idx, idx_to_char

class TextDataset(Dataset):
    def __init__(self, data_path, seq_len, char_to_idx):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.pad_token_id = self.char_to_idx.get('<pad>', 0)

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}. Dataset will be empty.")
            self.lines = []

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        tokenized_line = [self.char_to_idx.get(char, self.pad_token_id) for char in line]

        # Pad or truncate
        if len(tokenized_line) < self.seq_len:
            tokenized_line += [self.pad_token_id] * (self.seq_len - len(tokenized_line))
        else:
            tokenized_line = tokenized_line[:self.seq_len]

        return torch.tensor(tokenized_line, dtype=torch.long)

class MiniLLM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout, activation_function):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            activation=activation_function,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.transformer_encoder(x, mask=mask)
        x = self.fc(x)
        return x
