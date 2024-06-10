import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim, dtype=torch.float32)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.embedding(x)
