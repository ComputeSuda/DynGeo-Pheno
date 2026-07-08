#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from Bio.PDB import PDBParser
from tqdm import tqdm

# ======================= Config =======================
INPUT_CSV = "./preprocess/processed_samples_train.csv"
PDB_FOLDER = "./pdbs"
OUTPUT_FOLDER = "./features/contact_maps"
CUTOFF = 10.0  # Adjacency matrix cutoff distance (Angstrom)
# =====================================================

def parse_pdb_ca(pdb_file):
    coords = []
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ' and 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
    return np.array(coords)

def build_contact_map(pdb_path, cutoff=CUTOFF):
    coords = parse_pdb_ca(pdb_path)
    if len(coords) == 0:
        return None
    
    dist_matrix = squareform(pdist(coords))
    adjacency = (dist_matrix < cutoff).astype(np.float32)
    np.fill_diagonal(adjacency, 0)
    
    # Add self-loops
    contact_map = adjacency + np.identity(len(coords), dtype=np.float32)
    return contact_map

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    unique_proteins = df['UniProtKB_AC-ID'].unique()
    
    for pid in tqdm(unique_proteins, desc="Step 4: Constructing Structural Graphs"):
        out_path = os.path.join(OUTPUT_FOLDER, f"{pid}.npz")
        if os.path.exists(out_path):
            continue
            
        pdb_path = os.path.join(PDB_FOLDER, f"AF-{pid}-F1-model_v4.pdb")
        if not os.path.exists(pdb_path):
            continue
            
        cmap = build_contact_map(pdb_path)
        if cmap is not None:
            np.savez_compressed(out_path, C_alpha=cmap)

    print("Step 4 complete! Structural contact maps constructed.")

if __name__ == "__main__":
    main()
