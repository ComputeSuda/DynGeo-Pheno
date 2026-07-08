#!/usr/bin/env python3
import os
import numpy as np
from prody import parsePDB, ANM
import pandas as pd
from tqdm import tqdm

# ======================= Config =======================
INPUT_CSV = "./preprocess/processed_samples_train.csv"
PDB_FOLDER = "./pdbs"
OUTPUT_FOLDER_FRONT = "./features/anm_front_90d"
OUTPUT_FOLDER_LAST = "./features/anm_last_90d"
# =====================================================

def compute_anm_embeddings(pdb_path, n_modes=30):
    ubi = parsePDB(pdb_path)
    calphas = ubi.select('calpha and chain A')
    if calphas is None:
        calphas = ubi.select('calpha')
        
    anm = ANM('ANM analysis')
    anm.buildHessian(calphas)
    num_nodes = len(calphas)
    
    # Extract first 30 modes (low-frequency, global motions)
    anm.calcModes(n_modes=n_modes)
    vecs_front = anm.getEigvecs()  # shape: (3N, 30)
    
    # Extract last 30 modes (high-frequency, local rigidity)
    anm.calcModes(n_modes='all')
    vecs_all = anm.getEigvecs()
    vecs_last = vecs_all[:, -n_modes:]  # shape: (3N, 30)

    def flatten_to_nodes(vecs):
        embeds = []
        for i in range(num_nodes):
            vec = vecs[3*i:3*i+3, :]  # (3, 30)
            embeds.append(vec.flatten())  # (90,)
        return np.array(embeds)

    return flatten_to_nodes(vecs_front), flatten_to_nodes(vecs_last)

def main():
    os.makedirs(OUTPUT_FOLDER_FRONT, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER_LAST, exist_ok=True)
    
    df = pd.read_csv(INPUT_CSV)
    unique_proteins = df['UniProtKB_AC-ID'].unique()
    
    for pid in tqdm(unique_proteins, desc="Step 3: Extracting ANM Dynamic Features"):
        out_front = os.path.join(OUTPUT_FOLDER_FRONT, f"{pid}.npy")
        out_last = os.path.join(OUTPUT_FOLDER_LAST, f"{pid}.npy")
        if os.path.exists(out_front) and os.path.exists(out_last):
            continue
            
        pdb_path = os.path.join(PDB_FOLDER, f"AF-{pid}-F1-model_v4.pdb")
        if not os.path.exists(pdb_path):
            continue
            
        try:
            front_embed, last_embed = compute_anm_embeddings(pdb_path)
            np.save(out_front, front_embed)
            np.save(out_last, last_embed)
        except Exception as e:
            print(f"Failed to process {pid}: {e}")

    print("Step 3 complete! ANM 90-dim dynamic features extracted.")

if __name__ == "__main__":
    main()
