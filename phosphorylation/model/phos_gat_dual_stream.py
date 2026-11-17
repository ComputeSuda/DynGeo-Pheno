#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module: phos_gat_dual_stream.py
Description:
Macro-Micro Dual-Stream Graph Attention Network (PhosDualStreamGNN) for phosphorylation prediction :
1. Sequence projection layer (seq_proj): maps 128-d ESM embeddings to hidden_dim.
2. Adaptive frequency-band gating (FrequencyBandGating) and residual sequence-dynamics fusion.
3. Full-length sparse message passing over protein contact graphs.
4. Dual-stream readout combining global protein pooling (h_macro) and local site microenvironment pooling (h_micro).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class FrequencyBandGating(nn.Module):
    """Adaptive gating unit for low- and high-frequency dynamic features."""
    def __init__(self, in_dim_low, in_dim_high, out_dim):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim_low + in_dim_high, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.proj_low = nn.Linear(in_dim_low, out_dim)
        self.proj_high = nn.Linear(in_dim_high, out_dim)
        self.out_ln = nn.LayerNorm(out_dim)

    def forward(self, x_low, x_high):
        cat_feat = torch.cat([x_low, x_high], dim=-1)
        gate = self.gate_mlp(cat_feat)
        h_low = self.proj_low(x_low)
        h_high = self.proj_high(x_high)
        fused = gate * h_low + (1.0 - gate) * h_high
        return self.out_ln(fused), gate


class PhosDualStreamGNN(nn.Module):
    def __init__(
        self,
        dyn_dim=0,
        in_dim_low=0,
        in_dim_high=0,
        edge_dim=5,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_dual_band = (in_dim_low > 0 and in_dim_high > 0)
        self.use_dyn = (dyn_dim > 0 or self.is_dual_band)

        if hidden_dim == 128:
            self.seq_proj = nn.Identity()
        else:
            self.seq_proj = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

        # 1. Node dynamic adaptor and adaptive gating 
        if self.is_dual_band:
            self.band_gate = FrequencyBandGating(in_dim_low, in_dim_high, hidden_dim)
            self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        elif dyn_dim > 0:
            self.dyn_adaptor = nn.Sequential(
                nn.Linear(dyn_dim, 64),
                nn.GELU(),
                nn.Linear(64, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        self.node_ln = nn.LayerNorm(hidden_dim)

        # 2. Sparse GATv2 backbone 
        self.conv_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout
            )
            self.conv_layers.append(conv)
            self.ln_layers.append(nn.LayerNorm(hidden_dim))

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 3. Target site microenvironment attention pooling 
        self.micro_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # 4. Dual-stream classification head: [h_macro || h_micro] 
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, edge_index, edge_attr, site_masks, x_dyn=None, x_dyn_low=None, x_dyn_high=None):
        x_seq = self.seq_proj(x_seq)

        # Step 1: Node-level dynamic gating fusion 
        if self.is_dual_band and x_dyn_low is not None and x_dyn_high is not None:
            h_dyn, _ = self.band_gate(x_dyn_low, x_dyn_high)
            gate = torch.sigmoid(self.gate_fc(torch.cat([x_seq, h_dyn], dim=-1)))
            h = self.node_ln(x_seq + gate * h_dyn)
        elif self.use_dyn and x_dyn is not None and x_dyn.size(-1) > 0 and not torch.all(x_dyn == 0):
            h_dyn = self.dyn_adaptor(x_dyn)
            gate = torch.sigmoid(self.gate_fc(torch.cat([x_seq, h_dyn], dim=-1)))
            h = self.node_ln(x_seq + gate * h_dyn)
        else:
            h = self.node_ln(x_seq)

        # Step 2: Sparse full-protein message passing 
        for conv, ln in zip(self.conv_layers, self.ln_layers):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = ln(h)
            h = self.act(h)
            h = self.dropout(h)
            h = h + h_res

        # Step 3: Macro stream - global mean pooling across full protein 
        h_macro = h.mean(dim=0, keepdim=True)

        # Step 4: Micro stream and site-level scoring 
        logits_list = []
        for mask in site_masks:
            sub_h = h[mask]
            if sub_h.size(0) == 0:
                h_micro = h_macro
            else:
                attn_w = F.softmax(self.micro_attn(sub_h), dim=0)
                h_micro = (attn_w * sub_h).sum(dim=0, keepdim=True)

            h_pair = torch.cat([h_macro, h_micro], dim=-1)
            logit = self.classifier(h_pair)
            logits_list.append(logit.squeeze())

        return torch.stack(logits_list, dim=0) if len(logits_list) > 1 else logits_list[0].unsqueeze(0)
