#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: build_mutation_libraryA_graphs.py
Description:
1. Reads Dataset_A_Mutation_Model_TrainVal.xlsx.
2. Loads full-spectrum Tier-2 dynamic features (RMSF, CC, PRS, Stiffness, Invariants P0).
3. Constructs local 3D microenvironment Ego-Graphs (k=32, d <= 12.0 Å) for each missense variant.
4. Exports the superset graph cache dataset_A_mutation_graphs.pt.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch

# ======================= Path Configuration =======================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DATA_DIR = REPO_ROOT / "data"

SPECTRAL_DIR = BASE_DATA_DIR / "Raw_Spectral_Cache"
FEATURE_DIR  = BASE_DATA_DIR / "Extracted_Features"
ESM_WT_DIR   = BASE_DATA_DIR / "ESM_128d"
ESM_MUT_DIR  = BASE_DATA_DIR / "ESM_128d_mut"

MUT_DIR = Path(__file__).resolve().parent.parent / "processed_graphs"
INPUT_FILE_A = MUT_DIR / "Dataset_A_Mutation_Model_TrainVal.xlsx"
OUTPUT_CACHE_FILE = MUT_DIR / "dataset_A_mutation_graphs.pt"
# ==================================================================


def safe_npz_get(npz_data, key, fallback_key=None, default_val=0.0):
    """Safely retrieve array from NPZ archive with optional fallback."""
    if key in npz_data:
        return npz_data[key].squeeze()
    if fallback_key is not None and fallback_key in npz_data:
        return npz_data[fallback_key].squeeze()
    first_v = next(iter(npz_data.values()))
    return np.full(first_v.shape[0], default_val, dtype=np.float32)


def find_protein_files(acc_id, pdb_stem=None):
    spec_matches = list(SPECTRAL_DIR.glob(f"*{acc_id}*_spectral.npz"))
    if not spec_matches and pdb_stem:
        spec_matches = list(SPECTRAL_DIR.glob(f"{pdb_stem}_spectral.npz"))
    if not spec_matches:
        return None, "spectral_npz_missing"

    stem = spec_matches[0].name.replace("_spectral.npz", "")

    files = {
        'stem': stem,
        'spec': spec_matches[0],
        'rmsf': FEATURE_DIR / "RMSF" / f"{stem}_rmsf.npz",
        'cc': FEATURE_DIR / "CC" / f"{stem}_cc.npz",
        'prs': FEATURE_DIR / "PRS" / f"{stem}_prs.npz",
        'stiff': FEATURE_DIR / "Stiffness" / f"{stem}_stiffness.npz",
        'inv': FEATURE_DIR / "Invariants_P0" / f"{stem}_invariants.npz"
    }

    for p in [ESM_WT_DIR / f"{stem}.npy", ESM_WT_DIR / f"{acc_id}.npy", ESM_WT_DIR / f"AF-{acc_id}-F1-model_v4.npy"]:
        if p.is_file():
            files['esm_wt'] = p
            break
    if 'esm_wt' not in files:
        return None, "esm_wt_missing"

    for k in ['rmsf', 'cc', 'prs', 'stiff', 'inv']:
        if not files[k].exists():
            return None, f"missing_{k}"
    return files, "ok"


def find_mutation_esm(stem, acc_id, site, mutation_str):
    mut_clean = str(mutation_str).strip()
    for p in [
        ESM_MUT_DIR / f"{stem}_pos{site}_{mut_clean}.npy",
        ESM_MUT_DIR / f"{stem}_{mut_clean}.npy",
        ESM_MUT_DIR / f"AF-{acc_id}-F1-model_v4_pos{site}_{mut_clean}.npy",
        ESM_MUT_DIR / f"{acc_id}_pos{site}_{mut_clean}.npy",
    ]:
        if p.is_file():
            return p
    return None


def build_protein_mutation_records(acc_id, group_df):
    pdb_stem = group_df.iloc[0].get('pdb_stem', None)
    files, status = find_protein_files(acc_id, pdb_stem)
    if files is None:
        return None, status

    stem = files['stem']
    spec = np.load(files['spec'])
    coords, resnums = spec['coords'], spec['resnums']
    N = coords.shape[0]
    if N < 60 or N > 2700:
        return None, "length_out_of_range"

    esm_wt = np.load(files['esm_wt'])[:N, :]
    rmsf = np.load(files['rmsf'])
    cc = np.load(files['cc'])
    prs = np.load(files['prs'])
    stiff = np.load(files['stiff'])
    inv = np.load(files['inv'])

    node_dyn_dict = {
        # ANM RMSF
        'rmsf_anm_first10': safe_npz_get(rmsf, 'anm_first10_minmax', 'anm_first10'),
        'rmsf_anm_first20': safe_npz_get(rmsf, 'anm_first20_minmax', 'anm_first20'),
        'rmsf_anm_first30': safe_npz_get(rmsf, 'anm_first30_minmax', 'anm_first30'),
        'rmsf_anm_first40': safe_npz_get(rmsf, 'anm_first40_minmax', 'anm_first40'),
        'rmsf_anm_first60': safe_npz_get(rmsf, 'anm_first60_minmax', 'anm_first60'),
        'rmsf_anm_first2pct': safe_npz_get(rmsf, 'anm_first2pct_minmax', 'anm_first2pct', default_val=0.0),
        'rmsf_anm_all':     safe_npz_get(rmsf, 'anm_all_minmax', 'anm_all'),
        'rmsf_anm_last10':  safe_npz_get(rmsf, 'anm_last10_minmax', 'anm_last10'),
        'rmsf_anm_last20':  safe_npz_get(rmsf, 'anm_last20_minmax', 'anm_last20'),
        'rmsf_anm_last30':  safe_npz_get(rmsf, 'anm_last30_minmax', 'anm_last30'),
        'rmsf_anm_last40':  safe_npz_get(rmsf, 'anm_last40_minmax', 'anm_last40'),
        'rmsf_anm_last60':  safe_npz_get(rmsf, 'anm_last60_minmax', 'anm_last60'),
        'rmsf_anm_last2pct': safe_npz_get(rmsf, 'anm_last2pct_minmax', 'anm_last2pct', default_val=0.0),

        # GNM RMSF
        'rmsf_gnm_first10': safe_npz_get(rmsf, 'gnm_first10_minmax', 'gnm_first10'),
        'rmsf_gnm_first20': safe_npz_get(rmsf, 'gnm_first20_minmax', 'gnm_first20'),
        'rmsf_gnm_first30': safe_npz_get(rmsf, 'gnm_first30_minmax', 'gnm_first30'),
        'rmsf_gnm_first40': safe_npz_get(rmsf, 'gnm_first40_minmax', 'gnm_first40'),
        'rmsf_gnm_first60': safe_npz_get(rmsf, 'gnm_first60_minmax', 'gnm_first60'),
        'rmsf_gnm_first2pct': safe_npz_get(rmsf, 'gnm_first2pct_minmax', 'gnm_first2pct', default_val=0.0),
        'rmsf_gnm_all':     safe_npz_get(rmsf, 'gnm_all_minmax', 'gnm_all'),
        'rmsf_gnm_last10':  safe_npz_get(rmsf, 'gnm_last10_minmax', 'gnm_last10'),
        'rmsf_gnm_last20':  safe_npz_get(rmsf, 'gnm_last20_minmax', 'gnm_last20'),
        'rmsf_gnm_last30':  safe_npz_get(rmsf, 'gnm_last30_minmax', 'gnm_last30'),
        'rmsf_gnm_last40':  safe_npz_get(rmsf, 'gnm_last40_minmax', 'gnm_last40'),
        'rmsf_gnm_last60':  safe_npz_get(rmsf, 'gnm_last60_minmax', 'gnm_last60'),
        'rmsf_gnm_last2pct': safe_npz_get(rmsf, 'gnm_last2pct_minmax', 'gnm_last2pct', default_val=0.0),

    }

    resnums_list = list(resnums)
    samples = []

    for _, row in group_df.iterrows():
        site = int(row['site'])
        if site not in resnums_list:
            continue
        c_idx = resnums_list.index(site)
        mut_str = str(row.get('site_str', row.get('Mutation', f"{site}")))

        mut_f = find_mutation_esm(stem, acc_id, site, mut_str)
        if mut_f is not None:
            esm_mut = np.load(mut_f)[:N, :]
            esm_diff = (esm_wt - esm_mut).astype(np.float32)
        else:
            esm_diff = np.zeros_like(esm_wt, dtype=np.float32)

        dists = np.linalg.norm(coords - coords[c_idx], axis=1)
        sub_nodes = [i for i in np.argsort(dists)[:min(32, N)] if dists[i] <= 12.0]
        if c_idx not in sub_nodes:
            sub_nodes.insert(0, c_idx)
        sub_nodes = np.array(sub_nodes, dtype=int)
        M = len(sub_nodes)
        if M < 3:
            continue

        c_local = int(np.where(sub_nodes == c_idx)[0][0])
        sub_dist = cdist(coords[sub_nodes], coords[sub_nodes])
        adj = (sub_dist <= 10.0) & (sub_dist > 0.0)
        src, dst = np.where(adj)
        if len(src) == 0:
            src, dst = np.where(~np.eye(M, dtype=bool))

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        orig_s, orig_d = sub_nodes[src], sub_nodes[dst]

        d_vals = sub_dist[src, dst].astype(np.float32)
        inv_d = (1.0 / (d_vals + 1e-6)).astype(np.float32)

        def _get_edge_feature(npz_d, key, fallback_key=None):
            if key in npz_d:
                return npz_d[key][orig_s, orig_d].astype(np.float32)
            if fallback_key and fallback_key in npz_d:
                return npz_d[fallback_key][orig_s, orig_d].astype(np.float32)
            return np.zeros_like(d_vals, dtype=np.float32)

        edge_attr_dict = {
            'geo_dist':     d_vals,
            'inv_dist':     inv_d,
            'stiffness':    stiff['stiffness_norm'][orig_s, orig_d].astype(np.float32),
            'cc_first10':   _get_edge_feature(cc, 'cc_first10_norm'),
            'cc_first20':   _get_edge_feature(cc, 'cc_first20_norm'),
            'cc_first30':   _get_edge_feature(cc, 'cc_first30_norm'),
            'cc_first40':   _get_edge_feature(cc, 'cc_first40_norm'),
            'cc_first60':   _get_edge_feature(cc, 'cc_first60_norm'),
            'cc_first2pct': _get_edge_feature(cc, 'cc_first2pct_norm'),
            'cc_all':       _get_edge_feature(cc, 'cc_all_norm'),
            'cc_last10':    _get_edge_feature(cc, 'cc_last10_norm'),
            'cc_last20':    _get_edge_feature(cc, 'cc_last20_norm'),
            'cc_last30':    _get_edge_feature(cc, 'cc_last30_norm'),
            'cc_last40':    _get_edge_feature(cc, 'cc_last40_norm'),
            'cc_last60':    _get_edge_feature(cc, 'cc_last60_norm'),
            'cc_last2pct':  _get_edge_feature(cc, 'cc_last2pct_norm'),
        }

        samples.append({
            'mutation_id': f"{acc_id}_{mut_str}",
            'acc_id': acc_id,
            'site': site,
            'center_local_idx': c_local,
            'label': float(row['Label']),
            'fold': int(row.get('fold', -1)),
            'x_seq_diff': torch.tensor(esm_diff[sub_nodes], dtype=torch.float32),
            'x_seq_wt': torch.tensor(esm_wt[sub_nodes], dtype=torch.float32),
            'node_dyn_dict': {k: torch.tensor(v[sub_nodes], dtype=torch.float32) for k, v in node_dyn_dict.items()},
            'edge_index': edge_index,
            'edge_attr_dict': {k: torch.tensor(v, dtype=torch.float32) for k, v in edge_attr_dict.items()},
            'num_nodes': M
        })

    return samples, "ok"


def main():
    print("=== [Mutation Super-Graph Builder] Constructing Ego-Graph Cache (Library A) ===")
    df = pd.read_excel(INPUT_FILE_A)
    all_graphs = []
    fail_reasons = {}

    for acc_id, group in tqdm(df.groupby('ACC_ID'), desc="Building Library A Graphs"):
        try:
            recs, status = build_protein_mutation_records(acc_id, group)
            if recs:
                all_graphs.extend(recs)
            else:
                fail_reasons[status] = fail_reasons.get(status, 0) + 1
        except Exception as e:
            err_type = type(e).__name__
            fail_reasons[f"EXC_{err_type}"] = fail_reasons.get(f"EXC_{err_type}", 0) + 1
            continue

    MUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(all_graphs, OUTPUT_CACHE_FILE)
    print(f"\nGraph cache construction completed! Valid samples: {len(all_graphs):,}, saved to: {OUTPUT_CACHE_FILE}")
    if fail_reasons:
        print(f"  Skipped summary: {fail_reasons}")


if __name__ == "__main__":
    main()
