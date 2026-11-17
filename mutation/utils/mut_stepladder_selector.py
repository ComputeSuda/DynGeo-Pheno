#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module: mut_stepladder_selector.py
Description:
Dynamically assembles node dynamic features (x_dyn) and edge attributes (edge_attr)
for mutation Ego-Graphs across ablation presets and dual-band gating configurations.
"""

import torch

MUTATION_EXPERIMENT_PRESETS = {
    "exp_dyn_only": {
        "use_seq": False,
        "node_low":  ["rmsf_anm_first30", "rmsf_gnm_first30",
                      "prs_sens_first30", "prs_eff_first30"],
        "node_high": ["rmsf_anm_last30", "rmsf_gnm_last30",
                      "prs_sens_last30", "prs_eff_last30"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first30", "cc_last30", "stiffness"],
        "desc": "Dynamics-only baseline: masks ESM sequence features and relies strictly on physical dynamics"
    },
    "exp_0_seq_only": {
        "node_keys": [],
        "edge_keys": ["geo_dist", "inv_dist"],
        "desc": "ESM sequence-only baseline with geometric distance edges"
    },
    "exp_1a_rmsf_only": {
        "node_keys": ["rmsf_anm_first30", "rmsf_gnm_first30"],
        "edge_keys": ["geo_dist", "inv_dist"],
        "desc": "RMSF flexibility features only (ANM+GNM)"
    },
    "exp_1c_cc_only": {
        "node_keys": [],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first30"],
        "desc": "Cross-correlation edge bias only (CC_30)"
    },
    "exp_1d_stiff_only": {
        "node_keys": [],
        "edge_keys": ["geo_dist", "inv_dist", "stiffness"],
        "desc": "Mechanical stiffness edge constraint only"
    },
    "exp_2a_modes_10": {
        "node_keys": ["rmsf_anm_first10", "rmsf_gnm_first10"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first10", "stiffness"],
        "desc": "First 10 low-frequency modes"
    },
    "exp_2b_modes_20": {
        "node_keys": ["rmsf_anm_first20", "rmsf_gnm_first20"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first20", "stiffness"],
        "desc": "First 20 low-frequency modes"
    },
    "exp_2c_modes_30": {
        "node_keys": ["rmsf_anm_first30", "rmsf_gnm_first30"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first30", "stiffness"],
        "desc": "First 30 low-frequency modes"
    },
    "exp_2d_modes_40": {
        "node_keys": ["rmsf_anm_first40", "rmsf_gnm_first40"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first40", "stiffness"],
        "desc": "First 40 low-frequency modes"
    },
    "exp_2e_modes_60": {
        "node_keys": ["rmsf_anm_first60", "rmsf_gnm_first60"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first60", "stiffness"],
        "desc": "First 60 low-frequency modes"
    },
    "exp_2f_modes_2pct": {
        "node_keys": ["rmsf_anm_first2pct", "rmsf_gnm_first2pct"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first2pct", "stiffness"],
        "desc": "Top 2% low-frequency modes"
    },
    "exp_2g_modes_all": {
        "node_keys": ["rmsf_anm_all", "rmsf_gnm_all"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_all", "stiffness"],
        "desc": "All normal modes"
    },
    "exp_2h_modes_last10": {
        "node_keys": ["rmsf_anm_last10", "rmsf_gnm_last10"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_last10", "stiffness"],
        "desc": "Last 10 high-frequency modes"
    },
    "exp_2i_modes_last20": {
        "node_keys": ["rmsf_anm_last20", "rmsf_gnm_last20"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_last20", "stiffness"],
        "desc": "Last 20 high-frequency modes"
    },
    "exp_2j_modes_last30": {
        "node_keys": ["rmsf_anm_last30", "rmsf_gnm_last30"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_last30", "stiffness"],
        "desc": "Last 30 high-frequency modes"
    },
    "exp_2k_modes_last40": {
        "node_keys": ["rmsf_anm_last40", "rmsf_gnm_last40"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_last40", "stiffness"],
        "desc": "Last 40 high-frequency modes"
    },
    "exp_2l_modes_last60": {
        "node_keys": ["rmsf_anm_last60", "rmsf_gnm_last60"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_last60", "stiffness"],
        "desc": "Last 60 high-frequency modes"
    },
    "exp_2m_modes_last2pct": {
        "node_keys": ["rmsf_anm_last2pct", "rmsf_gnm_last2pct"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_last2pct", "stiffness"],
        "desc": "Last 2% high-frequency modes"
    },
    "exp_3a_mix_10": {
        "node_low": ["rmsf_anm_first10", "rmsf_gnm_first10"],
        "node_high": ["rmsf_anm_last10", "rmsf_gnm_last10"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first10", "cc_last10", "stiffness"],
        "desc": "Dual-band mix: first 10 + last 10 modes"
    },
    "exp_3b_mix_20": {
        "node_low": ["rmsf_anm_first20", "rmsf_gnm_first20"],
        "node_high": ["rmsf_anm_last20", "rmsf_gnm_last20"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first20", "cc_last20", "stiffness"],
        "desc": "Dual-band mix: first 20 + last 20 modes"
    },
    "exp_3c_mix_30": {
        "node_low": ["rmsf_anm_first30", "rmsf_gnm_first30"],
        "node_high": ["rmsf_anm_last30", "rmsf_gnm_last30"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first30", "cc_last30", "stiffness"],
        "desc": "Dual-band mix: first 30 + last 30 modes (Primary configuration)"
    },
    "exp_3d_mix_40": {
        "node_low": ["rmsf_anm_first40", "rmsf_gnm_first40"],
        "node_high": ["rmsf_anm_last40", "rmsf_gnm_last40"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first40", "cc_last40", "stiffness"],
        "desc": "Dual-band mix: first 40 + last 40 modes"
    },
    "exp_3e_mix_60": {
        "node_low": ["rmsf_anm_first60", "rmsf_gnm_first60"],
        "node_high": ["rmsf_anm_last60", "rmsf_gnm_last60"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first60", "cc_last60", "stiffness"],
        "desc": "Dual-band mix: first 60 + last 60 modes"
    },
    "exp_3f_mix_2pct": {
        "node_low": ["rmsf_anm_first2pct", "rmsf_gnm_first2pct"],
        "node_high": ["rmsf_anm_last2pct", "rmsf_gnm_last2pct"],
        "edge_keys": ["geo_dist", "inv_dist", "cc_first2pct", "cc_last2pct", "stiffness"],
        "desc": "Dual-band mix: first 2% + last 2% modes"
    }
}


def _extract_tensors(keys, tensor_dict, M, device):
    """Extract and concatenate tensors by key list."""
    tensors = []
    for k in keys:
        if k not in tensor_dict:
            continue
        t = tensor_dict[k]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        tensors.append(t)
    if len(tensors) > 0:
        return torch.cat(tensors, dim=-1)
    return torch.zeros((M, 1), dtype=torch.float32, device=device)


def assemble_mutation_tensors(sample_record, exp_preset="exp_0_seq_only"):
    """Assemble single-stream node and edge dynamic tensors."""
    cfg = MUTATION_EXPERIMENT_PRESETS[exp_preset]
    M = int(sample_record['num_nodes'])
    device = sample_record['x_seq_diff'].device
    node_dyn_dict = sample_record['node_dyn_dict']
    edge_attr_dict = sample_record['edge_attr_dict']

    if "node_low" in cfg and "node_high" in cfg:
        x_low = _extract_tensors(cfg["node_low"], node_dyn_dict, M, device)
        x_high = _extract_tensors(cfg["node_high"], node_dyn_dict, M, device)
        x_dyn = torch.cat([x_low, x_high], dim=-1)
    else:
        node_keys = cfg.get("node_keys", [])
        x_dyn = _extract_tensors(node_keys, node_dyn_dict, M, device)

    edge_tensors = []
    for k in cfg["edge_keys"]:
        t = edge_attr_dict[k]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        edge_tensors.append(t)
    edge_attr = torch.cat(edge_tensors, dim=-1)

    x_seq = sample_record['x_seq_diff']
    return x_seq, x_dyn, edge_attr


def assemble_mutation_dual_band_tensors(sample_record, exp_preset="exp_3c_mix_30"):
    """Assemble decoupled low- and high-frequency tensors for dual-band gated GNNs."""
    cfg = MUTATION_EXPERIMENT_PRESETS[exp_preset]
    M = int(sample_record['num_nodes'])
    device = sample_record['x_seq_diff'].device
    node_dyn_dict = sample_record['node_dyn_dict']
    edge_attr_dict = sample_record['edge_attr_dict']

    if "node_low" in cfg and "node_high" in cfg:
        x_dyn_low = _extract_tensors(cfg["node_low"], node_dyn_dict, M, device)
        x_dyn_high = _extract_tensors(cfg["node_high"], node_dyn_dict, M, device)
    else:
        x_dyn = _extract_tensors(cfg.get("node_keys", []), node_dyn_dict, M, device)
        x_dyn_low = x_dyn
        x_dyn_high = x_dyn

    edge_tensors = []
    for k in cfg["edge_keys"]:
        t = edge_attr_dict[k]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        edge_tensors.append(t)
    edge_attr = torch.cat(edge_tensors, dim=-1)

    x_seq = sample_record['x_seq_diff']
    return x_seq, x_dyn_low, x_dyn_high, edge_attr
