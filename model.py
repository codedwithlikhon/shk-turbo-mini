
import os
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_path, seq_len):
        # A real implementation would involve a proper tokenizer and data processing.
        # For this placeholder, we'll just read the number of lines to simulate having a dataset.
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.num_lines = sum(1 for line in f)
        except FileNotFoundError:
            print(f"Warning: Data file not found at {data_path}. Creating dummy data.")
            self.num_lines = 1000 # Create dummy data if file doesn't exist.
            # Create a dummy file so the script doesn't crash
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w', encoding='utf-8') as f:
                for _ in range(self.num_lines):
                    f.write("This is a dummy line.\n")

        self.seq_len = seq_len

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        # In a real scenario, you would read the specific line, tokenize it,
        # and return a tensor of shape (seq_len).
        # Here we just return a random tensor as a placeholder.
        return torch.randint(0, 30522, (self.seq_len,))

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
