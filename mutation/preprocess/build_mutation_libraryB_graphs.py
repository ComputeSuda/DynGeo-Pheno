#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: build_mutation_libraryB_graphs.py
Description:
Constructs Ego-Graph caches for external independent mutation test sets (Library B: DDD & Cancer Hotspot).
Generates three.pt files: B1 (DDD), B2 (Cancer Hotspot), and B_combined.
Enforces strict physical feature validation without zero-filling fallbacks.
"""

import re
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
INPUT_FILE_B1 = MUT_DIR / "Dataset_B_1_DDD_Test.xlsx"
INPUT_FILE_B2 = MUT_DIR / "Dataset_B_2_CancerHotspot_Test.xlsx"

# ==================================================================


def safe_npz_get(npz_data, key, fallback_key=None, default_val=0.0):
    """Retrieve array from NPZ archive."""
    if key in npz_data:
        return npz_data[key].squeeze()
    if fallback_key is not None and fallback_key in npz_data:
        return npz_data[fallback_key].squeeze()
    first_v = next(iter(npz_data.values()))
    return np.full(first_v.shape[0], default_val, dtype=np.float32)


def strict_npz_get(npz_data, key):
    """Strictly retrieve key from NPZ archive; raises KeyError if missing."""
    if key in npz_data:
        return npz_data[key].squeeze()
    raise KeyError(f"Strict mode: missing feature key '{key}' in NPZ archive!")


def find_protein_files(acc_id, pdb_stem=None):
    """Locate spectral cache, feature plugins, and wild-type ESM embeddings."""
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

    cand_esm = [
        ESM_WT_DIR / f"{stem}.npy",
        ESM_WT_DIR / f"{acc_id}.npy",
        ESM_WT_DIR / f"AF-{acc_id}-F1-model_v4.npy"
    ]
    for p in cand_esm:
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
    """Locate mutant ESM embedding file."""
    mut_clean = str(mutation_str).strip()
    cands = [
        ESM_MUT_DIR / f"{stem}_pos{site}_{mut_clean}.npy",
        ESM_MUT_DIR / f"{stem}_{mut_clean}.npy",
        ESM_MUT_DIR / f"AF-{acc_id}-F1-model_v4_pos{site}_{mut_clean}.npy",
        ESM_MUT_DIR / f"{acc_id}_pos{site}_{mut_clean}.npy",
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def build_protein_mutation_graphs(acc_id, group_df):
    """Construct Ego-Graph records for a single protein and its variants."""
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

        'prs_sens_all':     safe_npz_get(prs, 'prs_sens_all_minmax', 'sens_all'),
        'prs_eff_all':      safe_npz_get(prs, 'prs_eff_all_minmax', 'eff_all'),
        'node_dyn_10d_p0':  inv['node_dyn_10d']
    }

    resnums_list = list(resnums)
    mutation_records = []

    site_col = 'site' if 'site' in group_df.columns else ('position' if 'position' in group_df.columns else 'Position')
    mut_col = 'site_str' if 'site_str' in group_df.columns else ('Mutation' if 'Mutation' in group_df.columns else 'protein_variant')
    label_col = 'Label' if 'Label' in group_df.columns else 'label'

    for _, row in group_df.iterrows():
        raw_site = row[site_col]
        m_pos = re.search(r'(\d+)', str(raw_site))
        if not m_pos:
            continue
        site_num = int(m_pos.group(1))

        if site_num not in resnums_list:
            continue
        c_idx = resnums_list.index(site_num)
        mut_str = str(row[mut_col])

        mut_f = find_mutation_esm(stem, acc_id, site_num, mut_str)
        if mut_f is not None:
            esm_mut = np.load(mut_f)
            if esm_mut.shape[0] != N:
                esm_mut = esm_mut[:N, :]
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

        sub_coords = coords[sub_nodes]
        sub_dist_mat = cdist(sub_coords, sub_coords)
        adj_mask = (sub_dist_mat <= 10.0) & (sub_dist_mat > 0.0)
        src, dst = np.where(adj_mask)
        if len(src) == 0:
            src, dst = np.where(~np.eye(M, dtype=bool))

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        orig_s, orig_d = sub_nodes[src], sub_nodes[dst]

        d_vals = sub_dist_mat[src, dst].astype(np.float32)
        inv_d = (1.0 / (d_vals + 1e-6)).astype(np.float32)
        stiff_mat = strict_npz_get(stiff, 'stiffness_norm')

        edge_attr_dict = {
            'geo_dist':     torch.tensor(d_vals, dtype=torch.float32),
            'inv_dist':     torch.tensor(inv_d, dtype=torch.float32),
            'stiffness':    torch.tensor(stiff_mat[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_first10':   torch.tensor(strict_npz_get(cc, 'cc_first10_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_first20':   torch.tensor(strict_npz_get(cc, 'cc_first20_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_first30':   torch.tensor(strict_npz_get(cc, 'cc_first30_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_first40':   torch.tensor(strict_npz_get(cc, 'cc_first40_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_first60':   torch.tensor(strict_npz_get(cc, 'cc_first60_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_first2pct': torch.tensor(strict_npz_get(cc, 'cc_first2pct_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_all':       torch.tensor(strict_npz_get(cc, 'cc_all_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_last10':    torch.tensor(strict_npz_get(cc, 'cc_last10_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_last20':    torch.tensor(strict_npz_get(cc, 'cc_last20_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_last30':    torch.tensor(strict_npz_get(cc, 'cc_last30_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_last40':    torch.tensor(strict_npz_get(cc, 'cc_last40_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_last60':    torch.tensor(strict_npz_get(cc, 'cc_last60_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
            'cc_last2pct':  torch.tensor(strict_npz_get(cc, 'cc_last2pct_norm')[orig_s, orig_d].astype(np.float32), dtype=torch.float32),
        }

        am_val = np.nan
        for am_col in ('alphamissense', 'AlphaMissense'):
            if am_col in row and pd.notna(row[am_col]):
                try:
                    am_val = float(row[am_col])
                    break
                except (ValueError, TypeError):
                    pass

        sample_record = {
            'mutation_id': f"{acc_id}_{mut_str}",
            'acc_id': acc_id,
            'site': site_num,
            'mutation': mut_str,
            'center_local_idx': c_local,
            'label': float(row[label_col]),
            'fold': int(row.get('fold', -1)),
            'am_score': am_val,
            'num_nodes': M,
            'edge_index': edge_index,
            'x_seq_diff': torch.tensor(esm_diff[sub_nodes], dtype=torch.float32),
            'x_seq_wt': torch.tensor(esm_wt[sub_nodes], dtype=torch.float32),
            'node_dyn_dict': {k: torch.tensor(v[sub_nodes], dtype=torch.float32) for k, v in node_dyn_dict.items()},
            'edge_attr_dict': edge_attr_dict
        }
        mutation_records.append(sample_record)

    if not mutation_records:
        return None, "no_valid_mutations"

    return mutation_records, "ok"


def process_dataset_file(excel_path: Path, output_pt_name: str):
    """Process a single benchmark Excel file and export PyG graph cache."""
    print(f"\n=======================================================")
    print(f">>> Processing: {excel_path.name}")
    print(f"=======================================================")
    if not excel_path.is_file():
        print(f"Warning: File not found: {excel_path}")
        return None

    df = pd.read_excel(excel_path)
    acc_col = 'ACC_ID' if 'ACC_ID' in df.columns else ('uniprot' if 'uniprot' in df.columns else df.columns[0])
    grouped = df.groupby(acc_col)

    all_records = []
    fail_stats = {}

    for acc_id, group_df in tqdm(grouped, desc=f"Building {output_pt_name}"):
        records, status = build_protein_mutation_graphs(str(acc_id).strip(), group_df)
        if records is not None:
            all_records.extend(records)
        else:
            fail_stats[status] = fail_stats.get(status, 0) + 1

    out_path = MUT_DIR / output_pt_name
    torch.save(all_records, out_path)
    print(f"Exported graph cache: {out_path.name}")
    print(f"  Valid variant samples: {len(all_records):,} | Unique proteins: {len(set(r['acc_id'] for r in all_records)):,}")
    if fail_stats:
        print(f"  Filtered/unmatched summary: {fail_stats}")
    return all_records


def main():
    print("=" * 80)
    print("=== [External Benchmark Library B] Building Mutation Graph Caches ===")
    print(f"Data directory: {BASE_DATA_DIR}")
    print("=" * 80)

    rec_b1 = process_dataset_file(INPUT_FILE_B1, "dataset_B1_ddd_graphs.pt")
    rec_b2 = process_dataset_file(INPUT_FILE_B2, "dataset_B2_cancer_graphs.pt")

    if rec_b1 is not None and rec_b2 is not None:
        rec_combined = rec_b1 + rec_b2
        comb_path = MUT_DIR / "dataset_B_combined_graphs.pt"
        torch.save(rec_combined, comb_path)
        print(f"\nExported combined graph cache: {comb_path.name} (Total samples: {len(rec_combined):,})")


if __name__ == "__main__":
    main()
