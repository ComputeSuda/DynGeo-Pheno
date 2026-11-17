#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: train_cv.py
Description:
1. Loads processed_graphs/dataset_A_trainval_graphs.pt graph cache.
2. Interfaces with utils.dyn_stepladder_selector for ablation slicing and dual-band gating.
3. Executes strict 10-Fold protein-disjoint cross-validation (1:2 negative undersampling in training,
   unbiased natural prevalence evaluation in validation).
4. Saves best checkpoint per fold based on validation AUPRC and exports 10-fold results to Excel.
"""

import sys
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model.phos_gat_dual_stream import PhosDualStreamGNN
from utils.dyn_stepladder_selector import (
    EXPERIMENT_PRESETS,
    assemble_graph_tensors,
    assemble_graph_dual_band_tensors
)

# ======================= Path & Device Configuration =======================
BASE_DIR = CURRENT_DIR
GRAPH_CACHE_FILE = BASE_DIR / "processed_graphs" / "dataset_A_trainval_graphs.pt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ===========================================================================


def set_seed(seed: int = 2026):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def forward_step(model, prot_record, exp_preset, is_dual_band):
    """Assemble input tensors and return a callable forward pass for target site masks."""
    edge_index = prot_record["edge_index"].to(DEVICE)
    if exp_preset == "exp_dyn_only":
        x_seq = torch.zeros_like(prot_record["x_seq"]).to(DEVICE)
    else:
        x_seq = prot_record["x_seq"].to(DEVICE)

    if is_dual_band:
        x_low, x_high, edge_attr = assemble_graph_dual_band_tensors(prot_record, exp_preset)
        x_low = x_low.to(DEVICE)
        x_high = x_high.to(DEVICE)
        edge_attr = edge_attr.to(DEVICE)
        return lambda site_masks: model(
            x_seq=x_seq,
            edge_index=edge_index,
            edge_attr=edge_attr,
            site_masks=site_masks,
            x_dyn_low=x_low,
            x_dyn_high=x_high
        )
    else:
        x_dyn, edge_attr = assemble_graph_tensors(prot_record, exp_preset)
        x_dyn = x_dyn.to(DEVICE)
        edge_attr = edge_attr.to(DEVICE)
        return lambda site_masks: model(
            x_seq=x_seq,
            edge_index=edge_index,
            edge_attr=edge_attr,
            site_masks=site_masks,
            x_dyn=x_dyn
        )


def evaluate_fold(model, val_proteins, exp_preset, is_dual_band):
    """Unbiased evaluation on validation fold at natural prevalence."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for prot_record in val_proteins:
            site_masks = [s["micro_mask"].to(DEVICE) for s in prot_record["val_sites"]]
            labels = [float(s["label"]) for s in prot_record["val_sites"]]
            if len(site_masks) == 0:
                continue

            runner = forward_step(model, prot_record, exp_preset, is_dual_band)
            logits = runner(site_masks)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.view(-1).float()

            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_preds.extend(probs)
            all_labels.extend(labels)

    y_true = np.array(all_labels, dtype=np.int64)
    y_score = np.array(all_preds, dtype=np.float64)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    auroc = roc_auc_score(y_true, y_score)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(rec_curve, prec_curve)

    pred_bin = (y_score >= 0.5).astype(int)
    f1 = f1_score(y_true, pred_bin, zero_division=0)
    acc = accuracy_score(y_true, pred_bin)
    precision = precision_score(y_true, pred_bin, zero_division=0)
    recall = recall_score(y_true, pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true, pred_bin)

    return auroc, auprc, f1, acc, precision, recall, mcc


def train_single_fold(fold_idx, all_graphs, exp_preset, args):
    """Train a single fold and return best validation metrics."""
    print(f"\n>>> [Fold {fold_idx}/9] Starting training...")

    train_data, val_data = [], []
    for acc_id, record in all_graphs.items():
        train_sites = [s for s in record["sites"] if s["fold"] != fold_idx]
        val_sites = [s for s in record["sites"] if s["fold"] == fold_idx]

        if len(train_sites) > 0:
            rec_t = record.copy()
            rec_t["train_sites"] = train_sites
            train_data.append(rec_t)

        if len(val_sites) > 0:
            rec_v = record.copy()
            rec_v["val_sites"] = val_sites
            val_data.append(rec_v)

    print(f"    Training proteins: {len(train_data)} | Validation proteins: {len(val_data)}")

    cfg = EXPERIMENT_PRESETS[exp_preset]
    sample_rec = next(iter(all_graphs.values()))
    is_dual_band = ("node_low" in cfg and "node_high" in cfg)

    if is_dual_band:
        s_low, s_high, s_edge = assemble_graph_dual_band_tensors(sample_rec, exp_preset)
        in_dim_low = s_low.size(-1)
        in_dim_high = s_high.size(-1)
        dyn_dim = 0
        edge_dim = s_edge.size(-1)
    else:
        s_dyn, s_edge = assemble_graph_tensors(sample_rec, exp_preset)
        in_dim_low = 0
        in_dim_high = 0
        dyn_dim = s_dyn.size(-1) if s_dyn is not None else 0
        edge_dim = s_edge.size(-1)

    model = PhosDualStreamGNN(
        dyn_dim=dyn_dim,
        in_dim_low=in_dim_low,
        in_dim_high=in_dim_high,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auprc = -1.0
    best_record = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0.0, 0
        random.shuffle(train_data)

        for prot_record in train_data:
            pos_sites = [s for s in prot_record["train_sites"] if float(s["label"]) == 1.0]
            neg_sites = [s for s in prot_record["train_sites"] if float(s["label"]) == 0.0]
            if len(pos_sites) == 0 and len(neg_sites) == 0:
                continue

            n_neg_sample = min(len(neg_sites), max(1, len(pos_sites) * 2))
            sampled_neg = random.sample(neg_sites, n_neg_sample) if neg_sites else []
            batch_sites = pos_sites + sampled_neg
            if len(batch_sites) == 0:
                continue

            site_masks = [s["micro_mask"].to(DEVICE) for s in batch_sites]
            labels = torch.tensor(
                [float(s["label"]) for s in batch_sites], dtype=torch.float32, device=DEVICE
            )

            optimizer.zero_grad()
            runner = forward_step(model, prot_record, exp_preset, is_dual_band)
            logits = runner(site_masks)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits = logits.view(-1)

            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        if n_steps > 0:
            scheduler.step()
        avg_loss = total_loss / max(n_steps, 1)

        (val_auroc, val_auprc, val_f1,
         val_acc, val_prec, val_rec, val_mcc) = evaluate_fold(model, val_data, exp_preset, is_dual_band)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_record = {
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_mcc": val_mcc,
                "best_epoch": epoch,
            }
            ckpt_path = CHECKPOINT_DIR / f"{exp_preset}_fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | "
                f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f} "
                f"(Best: {best_val_auprc:.4f})"
            )

    if best_record is None:
        best_record = {
            k: 0.0 for k in (
                "val_auroc", "val_auprc", "val_f1", "val_acc",
                "val_precision", "val_recall", "val_mcc"
            )
        }
        best_record["best_epoch"] = 0

    print(
        f"Fold {fold_idx} finished | Best AUPRC: {best_record['val_auprc']:.4f} "
        f"(AUROC: {best_record['val_auroc']:.4f})"
    )
    return best_record


def main():
    parser = argparse.ArgumentParser(description="10-Fold CV Training Pipeline for Phosphorylation Prediction")
    parser.add_argument("--exp_preset", type=str, default="exp_3c_mix_30",
                        choices=list(EXPERIMENT_PRESETS.keys()),
                        help="Ablation experiment preset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden channel dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GATv2 layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=2026, help="Global random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 65)
    print("=== Phosphorylation 10-Fold CV Training Pipeline ===")
    print(f"Preset: {args.exp_preset} -> {EXPERIMENT_PRESETS[args.exp_preset].get('desc', '')}")
    print(f"Device: {DEVICE}")
    print(f"Cache File: {GRAPH_CACHE_FILE}")
    print("=" * 65)

    if not GRAPH_CACHE_FILE.is_file():
        raise FileNotFoundError(f"Graph cache file not found: {GRAPH_CACHE_FILE}")

    all_graphs = torch.load(GRAPH_CACHE_FILE, map_location="cpu")
    print(f"Loaded valid proteins: {len(all_graphs):,}")

    fold_results = []
    for f in range(10):
        res = train_single_fold(f, all_graphs, args.exp_preset, args)
        res["fold"] = f
        fold_results.append(res)

    res_df = pd.DataFrame(fold_results)
    metric_cols = [
        "val_auroc", "val_auprc", "val_f1", "val_acc",
        "val_precision", "val_recall", "val_mcc"
    ]

    mean_row = res_df[metric_cols].mean().to_dict()
    std_row = res_df[metric_cols].std(ddof=1).to_dict()
    mean_row["fold"], mean_row["best_epoch"] = "Mean", "-"
    std_row["fold"], std_row["best_epoch"] = "Std", "-"

    res_df_final = pd.concat(
        [res_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )

    out_path = RESULTS_DIR / f"result_{args.exp_preset}_10fold.xlsx"
    res_df_final.to_excel(out_path, index=False)

    print("\n=================== 10-Fold CV Summary ===================")
    print(f"Preset: {args.exp_preset}")
    print(f"  Mean AUROC     : {mean_row['val_auroc']:.4f} ± {std_row['val_auroc']:.4f}")
    print(f"  Mean AUPRC     : {mean_row['val_auprc']:.4f} ± {std_row['val_auprc']:.4f}")
    print(f"  Mean F1        : {mean_row['val_f1']:.4f} ± {std_row['val_f1']:.4f}")
    print(f"  Mean Accuracy  : {mean_row['val_acc']:.4f} ± {std_row['val_acc']:.4f}")
    print(f"  Mean Precision : {mean_row['val_precision']:.4f} ± {std_row['val_precision']:.4f}")
    print(f"  Mean Recall    : {mean_row['val_recall']:.4f} ± {std_row['val_recall']:.4f}")
    print(f"  Mean MCC       : {mean_row['val_mcc']:.4f} ± {std_row['val_mcc']:.4f}")
    print("==========================================================")
    print(f"Saved 10-fold results to: {out_path}")


if __name__ == "__main__":
    main()
