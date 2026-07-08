#!/usr/bin/env python3
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# ======================= Config =======================
INPUT_CSV = "./preprocess/processed_samples_train.csv"
LABEL_COL = "Regulatory_Disease"  # Adjust based on your CSV

ESM_FOLDER = "./features/esm_128d"
ANM_FRONT_FOLDER = "./features/anm_front_90d"
ANM_LAST_FOLDER = "./features/anm_last_90d"
CMAP_FOLDER = "./features/contact_maps"

OUTPUT_DIR = "./dataset"
KNN_LEN = 49
P1_LEN = 300
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

def standardize_cmap(cmap, target_len, indices=None):
    """Extract subgraph (if indices provided) and pad/crop contact map to (target_len, target_len)."""
    if cmap is None:
        return np.zeros((target_len, target_len), dtype=np.float32)
    
    if indices is not None:
        sub_cmap = cmap[np.ix_(indices, indices)]
    else:
        sub_cmap = cmap
        
    if sub_cmap.shape[0] < target_len:
        pad_width = target_len - sub_cmap.shape[0]
        padded = np.pad(sub_cmap, ((0, pad_width), (0, pad_width)), 'constant')
        return padded.astype(np.float32)
    else:
        return sub_cmap[:target_len, :target_len].astype(np.float32)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    
    knn_esm_list, knn_anm_f_list, knn_anm_l_list, knn_cmap_list = [], [], [], []
    p1_esm_list, p1_anm_f_list, p1_anm_l_list, p1_cmap_list = [], [], [], []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step 5: Building Dataset"):
        pid = str(row['UniProtKB_AC-ID']).strip()
        label = int(row[LABEL_COL])
        
        try:
            indices = ast.literal_eval(row['knn49_index'])
            indices = [i - 1 for i in indices]  # Convert to 0-based indexing
        except:
            indices = []
            
        esm_path = os.path.join(ESM_FOLDER, f"{pid}.npy")
        anm_f_path = os.path.join(ANM_FRONT_FOLDER, f"{pid}.npy")
        anm_l_path = os.path.join(ANM_LAST_FOLDER, f"{pid}.npy")
        cmap_path = os.path.join(CMAP_FOLDER, f"{pid}.npz")
        
        esm_feat = np.load(esm_path) if os.path.exists(esm_path) else None
        anm_f_feat = np.load(anm_f_path) if os.path.exists(anm_f_path) else None
        anm_l_feat = np.load(anm_l_path) if os.path.exists(anm_l_path) else None
        cmap_data = np.load(cmap_path)['C_alpha'] if os.path.exists(cmap_path) else None
        
        # 1. KNN49 Subgraph Features
        knn_esm_sub = esm_feat[indices, :] if esm_feat is not None and len(indices) > 0 else None
        knn_anm_f_sub = anm_f_feat[indices, :] if anm_f_feat is not None and len(indices) > 0 else None
        knn_anm_l_sub = anm_l_feat[indices, :] if anm_l_feat is not None and len(indices) > 0 else None
        
        knn_esm_list.append(standardize_shape(knn_esm_sub, KNN_LEN, ESM_DIM))
        knn_anm_f_list.append(standardize_shape(knn_anm_f_sub, KNN_LEN, ANM_DIM))
        knn_anm_l_list.append(standardize_shape(knn_anm_l_sub, KNN_LEN, ANM_DIM))
        knn_cmap_list.append(standardize_cmap(cmap_data, KNN_LEN, indices))
        
        # 2. P1 Full Graph Features
        p1_esm_list.append(standardize_shape(esm_feat, P1_LEN, ESM_DIM))
        p1_anm_f_list.append(standardize_shape(anm_f_feat, P1_LEN, ANM_DIM))
        p1_anm_l_list.append(standardize_shape(anm_l_feat, P1_LEN, ANM_DIM))
        p1_cmap_list.append(standardize_cmap(cmap_data, P1_LEN))
        
        labels.append(label)

    dataset = {
        "knn_esm": np.stack(knn_esm_list),
        "knn_anm_front": np.stack(knn_anm_f_list),
        "knn_anm_last": np.stack(knn_anm_l_list),
        "knn_cmap": np.stack(knn_cmap_list),
        "p1_esm": np.stack(p1_esm_list),
        "p1_anm_front": np.stack(p1_anm_f_list),
        "p1_anm_last": np.stack(p1_anm_l_list),
        "p1_cmap": np.stack(p1_cmap_list),
        "labels": np.array(labels)
    }
    
    out_path = os.path.join(OUTPUT_DIR, "train_dataset.npz")
    np.savez_compressed(out_path, **dataset)
    print(f"Step 5 complete! Dataset saved to: {out_path}")

if __name__ == "__main__":
    main()
