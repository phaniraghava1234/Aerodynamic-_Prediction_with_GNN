"""Baseline GNN for AirfRANS flow field regression.

Encoder-processor-decoder with GraphSAGE message passing. Dropout layers in
the encoder and decoder MLPs double as the stochastic source for MC Dropout
uncertainty quantification later — no architecture change needed.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

IN_FEATURES = 8   # pos(2) + inlet_vel(2) + dist(1) + normals(2) + surf(1)
OUT_FEATURES = 4  # u, v, p/rho, nu_t


class BaselineGNN(nn.Module):
    def __init__(self, hidden=128, n_conv=3, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(IN_FEATURES, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(n_conv))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_conv))
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, OUT_FEATURES),
        )

    def forward(self, data):
        h = self.encoder(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h, data.edge_index)))
        return self.decoder(h)
