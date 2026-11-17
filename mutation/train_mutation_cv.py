#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: train_mutation_cv.py
Description:
1. Loads dataset_A_mutation_graphs.pt superset graph cache.
2. Interfaces with utils.mut_stepladder_selector for ablation presets and dual-band gating.
3. Trains model.mut_gat_ego.MutEgoGATBatch using mini-batch PyG DataLoader.
4. Executes strict 10-Fold protein-disjoint cross-validation.
5. Exports 10-fold evaluation metrics to Excel.
"""

import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd

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

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.mut_stepladder_selector import (
    MUTATION_EXPERIMENT_PRESETS,
    assemble_mutation_tensors,
    assemble_mutation_dual_band_tensors
)
from model.mut_gat_ego import MutEgoGATBatch

# ======================= Path & Device Configuration =======================
BASE_DIR = Path(__file__).resolve().parent
CACHE_FILE = BASE_DIR / "processed_graphs" / "dataset_A_mutation_graphs.pt"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ===========================================================================


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_sample_to_pyg_data(sample, exp_preset):
    """Construct PyG Data object supporting both dual-band and single-stream presets."""
    cfg = MUTATION_EXPERIMENT_PRESETS[exp_preset]
    M = int(sample['num_nodes'])
    c_idx = int(sample['center_local_idx'])

    center_mask = torch.zeros(M, dtype=torch.bool)
    center_mask[c_idx] = True

    if "node_low" in cfg and "node_high" in cfg:
        x_seq_diff, x_low, x_high, edge_attr = assemble_mutation_dual_band_tensors(sample, exp_preset)
        if not cfg.get("use_seq", True):
            x_seq_diff = torch.zeros_like(x_seq_diff)

        data = Data(
            x_seq=x_seq_diff,
            x_dyn_low=x_low,
            x_dyn_high=x_high,
            edge_index=sample['edge_index'],
            edge_attr=edge_attr,
            center_mask=center_mask,
            y=torch.tensor([float(sample['label'])], dtype=torch.float32),
            num_nodes=M
        )
    else:
        x_seq_diff, x_dyn, edge_attr = assemble_mutation_tensors(sample, exp_preset)
        if not cfg.get("use_seq", True):
            x_seq_diff = torch.zeros_like(x_seq_diff)

        data = Data(
            x_seq=x_seq_diff,
            x_dyn=x_dyn,
            edge_index=sample['edge_index'],
            edge_attr=edge_attr,
            center_mask=center_mask,
            y=torch.tensor([float(sample['label'])], dtype=torch.float32),
            num_nodes=M
        )

    return data


def evaluate_val_fold(model, val_loader):
    """Unbiased evaluation on full validation fold returning 7 standard metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in val_loader:
            batch_data = batch_data.to(DEVICE)
            logits = model(batch_data)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            labels = batch_data.y.cpu().numpy().tolist()

            if isinstance(probs, float):
                probs = [probs]
            if isinstance(labels, float):
                labels = [labels]

            all_preds.extend(probs)
            all_labels.extend(labels)

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    auroc = roc_auc_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(rec, prec)
    pred_bin = (y_pred >= 0.5).astype(int)
    f1 = f1_score(y_true, pred_bin, zero_division=0)
    acc = accuracy_score(y_true, pred_bin)
    precision = precision_score(y_true, pred_bin, zero_division=0)
    recall = recall_score(y_true, pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true, pred_bin)

    return auroc, auprc, f1, acc, precision, recall, mcc


def train_single_fold(fold_idx, all_samples, exp_preset, args):
    print(f"\n=======================================================")
    print(f">>> [Fold {fold_idx}/9] Starting training...")

    train_pool = [s for s in all_samples if s['fold'] != fold_idx]
    val_pool = [s for s in all_samples if s['fold'] == fold_idx]

    train_pos = [s for s in train_pool if s['label'] == 1.0]
    train_neg = [s for s in train_pool if s['label'] == 0.0]

    val_pyg_list = [convert_sample_to_pyg_data(s, exp_preset) for s in val_pool]
    val_loader = DataLoader(val_pyg_list, batch_size=args.batch_size * 2, shuffle=False)

    val_pos_cnt = sum(1 for s in val_pool if s['label'] == 1.0)
    val_neg_cnt = len(val_pool) - val_pos_cnt
    print(f"  Training pool: Pos {len(train_pos):,} / Neg {len(train_neg):,}")
    print(f"  Validation pool: Total {len(val_pyg_list):,} (Natural ratio 1:{val_neg_cnt/max(1, val_pos_cnt):.2f})")

    cfg = MUTATION_EXPERIMENT_PRESETS[exp_preset]
    sample_rec = all_samples[0]

    if "node_low" in cfg and "node_high" in cfg:
        _, sample_xlow, sample_xhigh, sample_eattr = assemble_mutation_dual_band_tensors(sample_rec, exp_preset)
        in_dim_low = sample_xlow.size(-1)
        in_dim_high = sample_xhigh.size(-1)
        dyn_dim = 0
    else:
        _, sample_xdyn, sample_eattr = assemble_mutation_tensors(sample_rec, exp_preset)
        in_dim_low = 0
        in_dim_high = 0
        dyn_dim = sample_xdyn.size(-1) if sample_xdyn is not None else 0

    edge_dim = sample_eattr.size(-1)

    model = MutEgoGATBatch(
        dyn_dim=dyn_dim,
        in_dim_low=in_dim_low,
        in_dim_high=in_dim_high,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auprc = -1.0
    best_metrics = {}
    patience = args.patience
    no_improve_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        n_sample_neg = min(len(train_neg), len(train_pos) * 2)
        epoch_neg = random.sample(train_neg, n_sample_neg)
        epoch_samples = train_pos + epoch_neg
        random.shuffle(epoch_samples)

        train_pyg_list = [convert_sample_to_pyg_data(s, exp_preset) for s in epoch_samples]
        train_loader = DataLoader(train_pyg_list, batch_size=args.batch_size, shuffle=True)

        for batch_data in train_loader:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        val_auroc, val_auprc, val_f1, val_acc, val_prec, val_rec, val_mcc = evaluate_val_fold(model, val_loader)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            no_improve_count = 0
            best_metrics = {
                'fold': fold_idx,
                'epoch': epoch,
                'val_auroc': val_auroc,
                'val_auprc': val_auprc,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_mcc': val_mcc
            }
            ckpt_path = CHECKPOINT_DIR / f"{exp_preset}_mut_fold{fold_idx}_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            flag = "[Best Saved]"
        else:
            no_improve_count += 1
            flag = f"(Patience: {no_improve_count}/{patience})"

        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Loss: {total_loss:.4f} | "
              f"Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f} | Val MCC: {val_mcc:.4f} {flag}")

        if no_improve_count >= patience:
            print(f"  Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"Fold {fold_idx} finished | Best AUPRC: {best_metrics['val_auprc']:.4f} (AUROC: {best_metrics['val_auroc']:.4f}, MCC: {best_metrics['val_mcc']:.4f})")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="10-Fold CV Training Pipeline for Pathogenic Missense Mutation Prediction")
    parser.add_argument("--exp_preset", type=str, default="exp_3c_mix_30",
                        choices=list(MUTATION_EXPERIMENT_PRESETS.keys()),
                        help="Ablation experiment preset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per fold")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden channel dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GATv2 layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout ratio")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="L2 weight decay")
    args = parser.parse_args()

    set_seed(2026)
    print("=================================================================")
    print("=== Missense Mutation 10-Fold CV Training Pipeline ===")
    print(f"Preset: {args.exp_preset} -> {MUTATION_EXPERIMENT_PRESETS[args.exp_preset]['desc']}")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr} | Max Epochs: {args.epochs}")
    print(f"Device: {DEVICE}")
    print(f"Cache File: {CACHE_FILE}")
    print("=================================================================")

    if not CACHE_FILE.is_file():
        raise FileNotFoundError(f"Mutation graph cache not found: {CACHE_FILE}")

    all_samples = torch.load(CACHE_FILE, map_location="cpu")
    print(f"Loaded mutation samples: {len(all_samples):,}")

    fold_results = []
    for f in range(10):
        res = train_single_fold(f, all_samples, args.exp_preset, args)
        fold_results.append(res)

    res_df = pd.DataFrame(fold_results)

    mean_row = res_df.mean(numeric_only=True).to_dict()
    mean_row['fold'] = 'Mean'
    std_row = res_df.std(numeric_only=True).to_dict()
    std_row['fold'] = 'Std'

    res_df_final = pd.concat([res_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    print("\n======================= 10-Fold CV Summary =======================")
    print(f"Preset: {args.exp_preset}")
    print(f"  Mean AUROC     : {mean_row['val_auroc']:.4f} ± {std_row['val_auroc']:.4f}")
    print(f"  Mean AUPRC     : {mean_row['val_auprc']:.4f} ± {std_row['val_auprc']:.4f}")
    print(f"  Mean F1        : {mean_row['val_f1']:.4f} ± {std_row['val_f1']:.4f}")
    print(f"  Mean Accuracy  : {mean_row['val_acc']:.4f} ± {std_row['val_acc']:.4f}")
    print(f"  Mean Precision : {mean_row['val_precision']:.4f} ± {std_row['val_precision']:.4f}")
    print(f"  Mean Recall    : {mean_row['val_recall']:.4f} ± {std_row['val_recall']:.4f}")
    print(f"  Mean MCC       : {mean_row['val_mcc']:.4f} ± {std_row['val_mcc']:.4f}")
    print("==================================================================")

    log_out = RESULTS_DIR / f"result_mut_{args.exp_preset}_10fold.xlsx"
    res_df_final.to_excel(log_out, index=False)
    print(f"Saved 10-fold results to: {log_out}")


if __name__ == "__main__":
    main()
