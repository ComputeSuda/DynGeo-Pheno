#!/usr/bin/env python3
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# ======================= Config =======================
INPUT_CSV = "./preprocess/processed_samples_train.csv"
LABEL_COL = "clinvar_pathogenicity"  # Label column name

ESM_FOLDER = "./features/esm_128d"
ANM_FRONT_FOLDER = "./features/anm_front_90d"
ANM_LAST_FOLDER = "./features/anm_last_90d"
CMAP_FOLDER = "./features/contact_maps"

OUTPUT_DIR = "./dataset"
TARGET_LEN = 49
ESM_DIM = 128
ANM_DIM = 90
# =====================================================

def standardize_shape(arr, target_len, dim):
    """Pad or crop feature matrix to (target_len, dim)."""
    if arr is None:
        return np.zeros((target_len, dim), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        return np.zeros((target_len, dim), dtype=np.float32)

    if arr.shape[0] < target_len:
        pad = np.zeros((target_len - arr.shape[0], dim), dtype=np.float32)
        return np.vstack((arr.astype(np.float32), pad))
    else:
        return arr[:target_len, :].astype(np.float32)

def standardize_cmap(cmap, target_len, indices):
    """Extract and pad/crop contact map to (target_len, target_len)."""
    if cmap is None:
        return np.zeros((target_len, target_len), dtype=np.float32)

    # Extract subgraph
    sub_cmap = cmap[np.ix_(indices, indices)]

    if sub_cmap.shape[0] < target_len:
        pad_width = target_len - sub_cmap.shape[0]
        padded = np.pad(sub_cmap, ((0, pad_width), (0, pad_width)), 'constant')
        return padded.astype(np.float32)
    else:
        return sub_cmap[:target_len, :target_len].astype(np.float32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # Handle column name variants for wild/mut amino acids
    wt_col = 'wild_aa' if 'wild_aa' in df.columns else ('wild_a' if 'wild_a' in df.columns else None)
    mut_col = 'mut_aa' if 'mut_aa' in df.columns else ('mut_a' if 'mut_a' in df.columns else None)

    esm_list, anm_f_list, anm_l_list, cmap_list, labels = [], [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step 5: Building Dataset"):
        pid = str(row['UniProtKB_AC-ID']).strip()
        position = int(row['position'])
        label = int(row[LABEL_COL])

        # Build mutation-specific ESM filename: {pid}_{wt}{pos}{mut}.npy
        if wt_col and mut_col:
            wt_aa = str(row[wt_col]).strip().upper()
            mut_aa = str(row[mut_col]).strip().upper()
            esm_name = f"{pid}_{wt_aa}{position}{mut_aa}.npy"
        else:
            # Fallback: try mutant_sequence column or protein-level embedding
            esm_name = f"{pid}.npy"

        # Parse knn49 indices (stored as string representation of list in CSV)
        try:
            indices = ast.literal_eval(row['knn49_index'])
            indices = [i - 1 for i in indices]  # Convert to 0-based indexing
        except:
            indices = []

        # 1. Load features
        #    ESM: mutation-specific (WT - Mut)
        #    ANM & CMap: protein-level (structural features of WT structure)
        esm_path = os.path.join(ESM_FOLDER, esm_name)
        anm_f_path = os.path.join(ANM_FRONT_FOLDER, f"{pid}.npy")
        anm_l_path = os.path.join(ANM_LAST_FOLDER, f"{pid}.npy")
        cmap_path = os.path.join(CMAP_FOLDER, f"{pid}.npz")

        esm_feat = np.load(esm_path) if os.path.exists(esm_path) else None
        anm_f_feat = np.load(anm_f_path) if os.path.exists(anm_f_path) else None
        anm_l_feat = np.load(anm_l_path) if os.path.exists(anm_l_path) else None
        cmap_data = np.load(cmap_path)['C_alpha'] if os.path.exists(cmap_path) else None

        # 2. Extract 49-residue subset
        esm_sub = esm_feat[indices, :] if esm_feat is not None and len(indices) > 0 else None
        anm_f_sub = anm_f_feat[indices, :] if anm_f_feat is not None and len(indices) > 0 else None
        anm_l_sub = anm_l_feat[indices, :] if anm_l_feat is not None and len(indices) > 0 else None
        cmap_sub = cmap_data[np.ix_(indices, indices)] if cmap_data is not None and len(indices) > 0 else None

        # 3. Standardize shapes
        esm_list.append(standardize_shape(esm_sub, TARGET_LEN, ESM_DIM))
        anm_f_list.append(standardize_shape(anm_f_sub, TARGET_LEN, ANM_DIM))
        anm_l_list.append(standardize_shape(anm_l_sub, TARGET_LEN, ANM_DIM))
        cmap_list.append(standardize_cmap(cmap_sub, TARGET_LEN, indices) if cmap_sub is not None else np.zeros((TARGET_LEN, TARGET_LEN), dtype=np.float32))
        labels.append(label)

    # 4. Stack and save
    final_esm = np.stack(esm_list)
    final_anm_f = np.stack(anm_f_list)
    final_anm_l = np.stack(anm_l_list)
    final_cmap = np.stack(cmap_list)
    final_labels = np.array(labels)

    out_path = os.path.join(OUTPUT_DIR, "train_dataset.npz")
    np.savez_compressed(
        out_path,
        esm=final_esm,
        anm_front=final_anm_f,
        anm_last=final_anm_l,
        cmap=final_cmap,
        labels=final_labels
    )

    print(f"Step 5 complete! Dataset saved to: {out_path}")
    print(f"ESM Shape: {final_esm.shape}, ANM_front Shape: {final_anm_f.shape}, CMAP Shape: {final_cmap.shape}")

if __name__ == "__main__":
    main()
