#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: build_phos_graphs.py

Description:
1. Reads phosphorylation benchmark tables (Library A for 10-Fold CV and
   Library B for external test).
2. Loads Tier-2 physical features (RMSF, CC, PRS, Stiffness, Invariants P0)
   and ESM-2 embeddings.
3. Constructs full-length protein contact graphs (d_ij <= 10.0 A) and local
   8.0 A site masks.
4. Exports superset graph caches dataset_A_trainval_graphs.pt and
   dataset_B_test_graphs.pt.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# ======================= Path Configuration =======================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DATA_DIR = REPO_ROOT / "data"
SPECTRAL_DIR = BASE_DATA_DIR / "Raw_Spectral_Cache"
FEATURE_DIR = BASE_DATA_DIR / "Extracted_Features"
ESM_DIR = BASE_DATA_DIR / "ESM_128d"
OUTPUT_GRAPH_DIR = Path(__file__).resolve().parent.parent / "processed_graphs"
PHOS_SCHEME1_DIR = OUTPUT_GRAPH_DIR / "data_scheme1"
OUTPUT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
# ==================================================================

# Module-level counter for missing feature keys (reset per protein build)
_MISSING_KEY_LOG = []


def safe_npz_get(npz_data, key, default_shape=None, default_val=0.0):
    """Retrieve array from NPZ archive; on missing key, fill with zeros and
    record the event (no cross-frequency-band substitution)."""
    if key in npz_data:
        return npz_data[key].squeeze()
    _MISSING_KEY_LOG.append(key)
    if default_shape is not None:
        return np.full(default_shape, default_val, dtype=np.float32)
    first_v = next(iter(npz_data.values()))
    return np.full(first_v.shape[0], default_val, dtype=np.float32)


def find_protein_files(acc_id, pdb_stem=None):
    """Locate spectral cache, dynamic plugins, and ESM embedding files."""
    spec_matches = list(SPECTRAL_DIR.glob(f"*{acc_id}*_spectral.npz"))
    if not spec_matches and pdb_stem:
        spec_matches = list(SPECTRAL_DIR.glob(f"{pdb_stem}_spectral.npz"))
    if not spec_matches:
        return None, "spectral_npz_missing"

    spec_path = spec_matches[0]
    stem = spec_path.name.replace("_spectral.npz", "")
    files = {
        'stem': stem,
        'spec': spec_path,
        'rmsf': FEATURE_DIR / "RMSF" / f"{stem}_rmsf.npz",
        'cc': FEATURE_DIR / "CC" / f"{stem}_cc.npz",
        'prs': FEATURE_DIR / "PRS" / f"{stem}_prs.npz",
        'stiff': FEATURE_DIR / "Stiffness" / f"{stem}_stiffness.npz",
        'inv': FEATURE_DIR / "Invariants_P0" / f"{stem}_invariants.npz",
    }
    esm_cands = [
        ESM_DIR / f"{acc_id}.npy",
        ESM_DIR / f"{stem}.npy",
        ESM_DIR / f"AF-{acc_id}-F1-model_v4.npy",
    ]
    esm_path = next((p for p in esm_cands if p.is_file()), None)
    if esm_path is None:
        return None, "esm_missing"
    files['esm'] = esm_path
    for k in ['rmsf', 'cc', 'prs', 'stiff', 'inv']:
        if not files[k].exists():
            return None, f"missing_{k}_{files[k].name}"
    return files, "ok"


def build_protein_super_graph(acc_id, site_group_df):
    pdb_stem = site_group_df.iloc[0].get('pdb_stem', None)
    files, status = find_protein_files(acc_id, pdb_stem)
    if files is None:
        return None, status

    _MISSING_KEY_LOG.clear()

    spec_data = np.load(files['spec'])
    coords = spec_data['coords']
    resnums = spec_data['resnums']
    N = coords.shape[0]
    if N < 60 or N > 2700:
        return None, f"length_out_of_range_{N}"

    esm_feat = np.load(files['esm'])
    if esm_feat.shape[0] != N:
        esm_feat = esm_feat[:N, :]

    # Contact graph (<= 10 A)
    dist_mat = squareform(pdist(coords))
    adj_mask = (dist_mat <= 10.0) & (dist_mat > 0.0)
    src, dst = np.where(adj_mask)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    d_vals = dist_mat[src, dst].astype(np.float32)
    inv_d = (1.0 / (d_vals + 1e-6)).astype(np.float32)

    rmsf_data = np.load(files['rmsf'])
    cc_data = np.load(files['cc'])
    prs_data = np.load(files['prs'])
    stiff_data = np.load(files['stiff'])
    inv_data = np.load(files['inv'])

    # ---- Node dynamics: exact key mapping, zeros when the band is unavailable ----
    node_dyn_dict = {
        # ANM RMSF
        'rmsf_anm_first10': safe_npz_get(rmsf_data, 'anm_first10_minmax'),
        'rmsf_anm_first20': safe_npz_get(rmsf_data, 'anm_first20_minmax'),
        'rmsf_anm_first30': safe_npz_get(rmsf_data, 'anm_first30_minmax'),
        'rmsf_anm_first40': safe_npz_get(rmsf_data, 'anm_first40_minmax'),
        'rmsf_anm_first60': safe_npz_get(rmsf_data, 'anm_first60_minmax'),
        'rmsf_anm_first2pct': safe_npz_get(rmsf_data, 'anm_first2pct_minmax'),
        'rmsf_anm_all': safe_npz_get(rmsf_data, 'anm_all_minmax'),
        'rmsf_anm_last10': safe_npz_get(rmsf_data, 'anm_last10_minmax'),
        'rmsf_anm_last20': safe_npz_get(rmsf_data, 'anm_last20_minmax'),
        'rmsf_anm_last30': safe_npz_get(rmsf_data, 'anm_last30_minmax'),
        'rmsf_anm_last40': safe_npz_get(rmsf_data, 'anm_last40_minmax'),
        'rmsf_anm_last60': safe_npz_get(rmsf_data, 'anm_last60_minmax'),
        'rmsf_anm_last2pct': safe_npz_get(rmsf_data, 'anm_last2pct_minmax'),
        # GNM RMSF
        'rmsf_gnm_first10': safe_npz_get(rmsf_data, 'gnm_first10_minmax'),
        'rmsf_gnm_first20': safe_npz_get(rmsf_data, 'gnm_first20_minmax'),
        'rmsf_gnm_first30': safe_npz_get(rmsf_data, 'gnm_first30_minmax'),
        'rmsf_gnm_first40': safe_npz_get(rmsf_data, 'gnm_first40_minmax'),
        'rmsf_gnm_first60': safe_npz_get(rmsf_data, 'gnm_first60_minmax'),
        'rmsf_gnm_first2pct': safe_npz_get(rmsf_data, 'gnm_first2pct_minmax'),
        'rmsf_gnm_all': safe_npz_get(rmsf_data, 'gnm_all_minmax'),
        'rmsf_gnm_last10': safe_npz_get(rmsf_data, 'gnm_last10_minmax'),
        'rmsf_gnm_last20': safe_npz_get(rmsf_data, 'gnm_last20_minmax'),
        'rmsf_gnm_last30': safe_npz_get(rmsf_data, 'gnm_last30_minmax'),
        'rmsf_gnm_last40': safe_npz_get(rmsf_data, 'gnm_last40_minmax'),
        'rmsf_gnm_last60': safe_npz_get(rmsf_data, 'gnm_last60_minmax'),
        'rmsf_gnm_last2pct': safe_npz_get(rmsf_data, 'gnm_last2pct_minmax'),
        # PRS Sensitivity & Effectiveness
        'prs_sens_first10': safe_npz_get(prs_data, 'prs_sens_first10_minmax'),
        'prs_eff_first10': safe_npz_get(prs_data, 'prs_eff_first10_minmax'),
        'prs_sens_first20': safe_npz_get(prs_data, 'prs_sens_first20_minmax'),
        'prs_eff_first20': safe_npz_get(prs_data, 'prs_eff_first20_minmax'),
        'prs_sens_first30': safe_npz_get(prs_data, 'prs_sens_first30_minmax'),
        'prs_eff_first30': safe_npz_get(prs_data, 'prs_eff_first30_minmax'),
        'prs_sens_first40': safe_npz_get(prs_data, 'prs_sens_first40_minmax'),
        'prs_eff_first40': safe_npz_get(prs_data, 'prs_eff_first40_minmax'),
        'prs_sens_first60': safe_npz_get(prs_data, 'prs_sens_first60_minmax'),
        'prs_eff_first60': safe_npz_get(prs_data, 'prs_eff_first60_minmax'),
        'prs_sens_last10': safe_npz_get(prs_data, 'prs_sens_last10_minmax'),
        'prs_eff_last10': safe_npz_get(prs_data, 'prs_eff_last10_minmax'),
        'prs_sens_last20': safe_npz_get(prs_data, 'prs_sens_last20_minmax'),
        'prs_eff_last20': safe_npz_get(prs_data, 'prs_eff_last20_minmax'),
        'prs_sens_last30': safe_npz_get(prs_data, 'prs_sens_last30_minmax'),
        'prs_eff_last30': safe_npz_get(prs_data, 'prs_eff_last30_minmax'),
        'prs_sens_last40': safe_npz_get(prs_data, 'prs_sens_last40_minmax'),
        'prs_eff_last40': safe_npz_get(prs_data, 'prs_eff_last40_minmax'),
        'prs_sens_last60': safe_npz_get(prs_data, 'prs_sens_last60_minmax'),
        'prs_eff_last60': safe_npz_get(prs_data, 'prs_eff_last60_minmax'),
        'prs_sens_first2pct': safe_npz_get(prs_data, 'prs_sens_first2pct_minmax'),
        'prs_eff_first2pct': safe_npz_get(prs_data, 'prs_eff_first2pct_minmax'),
        'prs_sens_last2pct': safe_npz_get(prs_data, 'prs_sens_last2pct_minmax'),
        'prs_eff_last2pct': safe_npz_get(prs_data, 'prs_eff_last2pct_minmax'),
        'prs_sens_all': safe_npz_get(prs_data, 'prs_sens_all_minmax'),
        'prs_eff_all': safe_npz_get(prs_data, 'prs_eff_all_minmax'),
        # SO(3) Spectral Invariants
        'node_dyn_10d_p0': inv_data['node_dyn_10d'],
    }

    def _get_edge_feature(npz_d, key):
        if key in npz_d:
            return npz_d[key][src, dst].astype(np.float32)
        _MISSING_KEY_LOG.append(key)
        return np.zeros_like(d_vals, dtype=np.float32)

    # ---- Edge attributes: exact keys, zeros on missing ----
    edge_attr_dict = {
        'geo_dist': d_vals,
        'inv_dist': inv_d,
        'stiffness': stiff_data['stiffness_norm'][src, dst].astype(np.float32),
        'cc_first10': _get_edge_feature(cc_data, 'cc_first10_norm'),
        'cc_first20': _get_edge_feature(cc_data, 'cc_first20_norm'),
        'cc_first30': _get_edge_feature(cc_data, 'cc_first30_norm'),
        'cc_first40': _get_edge_feature(cc_data, 'cc_first40_norm'),
        'cc_first60': _get_edge_feature(cc_data, 'cc_first60_norm'),
        'cc_first2pct': _get_edge_feature(cc_data, 'cc_first2pct_norm'),
        'cc_all': _get_edge_feature(cc_data, 'cc_all_norm'),
        'cc_last10': _get_edge_feature(cc_data, 'cc_last10_norm'),
        'cc_last20': _get_edge_feature(cc_data, 'cc_last20_norm'),
        'cc_last30': _get_edge_feature(cc_data, 'cc_last30_norm'),
        'cc_last40': _get_edge_feature(cc_data, 'cc_last40_norm'),
        'cc_last60': _get_edge_feature(cc_data, 'cc_last60_norm'),
        'cc_last2pct': _get_edge_feature(cc_data, 'cc_last2pct_norm'),
    }

    # Per-site records with 8 A micro-environment masks
    site_list = []
    resnums_list = list(resnums)
    for _, row in site_group_df.iterrows():
        pos = int(row['site'])
        if pos in resnums_list:
            node_idx = resnums_list.index(pos)
            micro_mask = (dist_mat[node_idx] <= 8.0)
            site_list.append({
                'site_id': str(row.get('site_str', f"{acc_id}_{pos}")),
                'acc_id': acc_id,
                'target_node_idx': node_idx,
                'micro_mask': torch.tensor(micro_mask, dtype=torch.bool),
                'label': float(row['Label']),
                'fold': int(row.get('fold', -1)),
            })

    if not site_list:
        return None, "no_valid_sites_in_structure"

    record = {
        'x_seq': torch.tensor(esm_feat, dtype=torch.float32),
        'node_dyn_dict': {k: torch.tensor(v, dtype=torch.float32)
                          for k, v in node_dyn_dict.items()},
        'edge_index': edge_index,
        'edge_attr_dict': {k: torch.tensor(v, dtype=torch.float32)
                           for k, v in edge_attr_dict.items()},
        'num_nodes': N,
        'sites': site_list,
    }

    fallback_note = f" | zero-filled keys: {sorted(set(_MISSING_KEY_LOG))}" if _MISSING_KEY_LOG else ""
    return record, "ok" + fallback_note


def process_dataset(excel_path, output_pt_name):
    print(f"\n=======================================================")
    print(f">>> Building full-protein dynamic superset graphs: {excel_path.name}")
    if not excel_path.is_file():
        print(f" Warning: File not found: {excel_path}")
        return

    df = pd.read_excel(excel_path)
    grouped = df.groupby('ACC_ID')
    protein_graphs = {}
    total_sites = 0
    fail_reasons = {}
    proteins_with_zero_fill = 0

    for acc_id, group_df in tqdm(grouped, desc=output_pt_name):
        try:
            rec, status = build_protein_super_graph(acc_id, group_df)
            if rec is not None:
                protein_graphs[acc_id] = rec
                total_sites += len(rec['sites'])
                if "zero-filled keys" in status:
                    proteins_with_zero_fill += 1
                    print(f" [WARN] {acc_id}: {status}")
            else:
                fail_reasons[status] = fail_reasons.get(status, 0) + 1
        except Exception as e:
            err_type = type(e).__name__
            fail_reasons[f"EXC_{err_type}"] = fail_reasons.get(f"EXC_{err_type}", 0) + 1
            continue

    out_file = OUTPUT_GRAPH_DIR / output_pt_name
    torch.save(protein_graphs, out_file)
    print(f"Saved superset graph cache: {out_file.name} "
          f"(Proteins: {len(protein_graphs):,} | Sites: {total_sites:,})")
    print(f" Proteins with zero-filled unavailable bands: {proteins_with_zero_fill:,}")
    if fail_reasons:
        print(f" Skipped/Failed summary: {fail_reasons}")


def main():
    print("=== [Phos Super-Graph Builder] Constructing Full-Protein Graph Caches ===")
    print(f"Data directory: {BASE_DATA_DIR}")
    file_a = PHOS_SCHEME1_DIR / "Dataset_A_PTMD2_Model_TrainVal.xlsx"
    file_b = PHOS_SCHEME1_DIR / "Dataset_B_External_Test_Orthogonal.xlsx"
    process_dataset(file_a, "dataset_A_trainval_graphs.pt")
    process_dataset(file_b, "dataset_B_test_graphs.pt")
    print("\nCompleted phosphorylation graph cache construction!")


if __name__ == "__main__":
    main()
