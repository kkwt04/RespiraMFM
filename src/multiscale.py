"""Multi-scale fusion modules for the optional RespiraMFM extension."""

import torch
import torch.nn as nn

class MultiScaleFusion(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, embedding_20, embedding_100, embedding_400):
        embeddings = (embedding_20, embedding_100, embedding_400)
        if any(item.ndim != 2 or item.shape[1] != self.embedding_dim for item in embeddings):
            raise ValueError(f"Expected three [batch, {self.embedding_dim}] embeddings")
        if len({item.shape[0] for item in embeddings}) != 1:
            raise ValueError("Multi-scale embeddings must have the same batch size")
        return self.network(torch.cat(embeddings, dim=1))

class MultiScaleProjectionModel(nn.Module):
    """Phase 1 trainable fusion followed by the existing aligner."""

    def __init__(self, fusion, aligner):
        super().__init__()
        self.fusion = fusion
        self.aligner = aligner

    def forward(self, embedding_20, embedding_100, embedding_400):
        return self.aligner(self.fusion(embedding_20, embedding_100, embedding_400))


def load_multiscale_checkpoint(fusion, aligner, checkpoint_path, freeze=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    fusion.load_state_dict(checkpoint["fusion_state_dict"])
    aligner.load_state_dict(checkpoint["aligner_state_dict"])
    if freeze:
        for module in (fusion, aligner):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
    return checkpoint
