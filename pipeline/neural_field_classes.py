"""Neural field model classes (shared between training and inference)."""
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, n_freqs=10, input_dim=3):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs).float()
        self.register_buffer("freqs", freqs)
        self.output_dim = input_dim + input_dim * 2 * n_freqs

    def forward(self, x):
        encoded = [x]
        for freq in self.freqs:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)


class NeuralField(nn.Module):
    def __init__(self, n_classes=13, n_freqs=10, hidden_dim=256, n_layers=5):
        super().__init__()
        self.encoder = PositionalEncoding(n_freqs=n_freqs)

        mid = n_layers // 2
        self.pre_skip = nn.ModuleList()
        self.post_skip = nn.ModuleList()

        enc_dim = self.encoder.output_dim
        for i in range(n_layers):
            if i < mid:
                self.pre_skip.append(nn.Linear(enc_dim if i == 0 else hidden_dim, hidden_dim))
            elif i == mid:
                self.post_skip.append(nn.Linear(hidden_dim + enc_dim, hidden_dim))
            else:
                self.post_skip.append(nn.Linear(hidden_dim, hidden_dim))

        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, coords):
        enc = self.encoder(coords)
        h = enc
        for layer in self.pre_skip:
            h = torch.relu(layer(h))
        h = torch.cat([h, enc], dim=-1)
        for layer in self.post_skip:
            h = torch.relu(layer(h))
        return self.head(h)
