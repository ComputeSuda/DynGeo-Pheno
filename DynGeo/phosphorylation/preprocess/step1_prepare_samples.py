#!/usr/bin/env python3
import os
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

# ======================= Config =======================
INPUT_CSV = "./example/train.csv"
PDB_FOLDER = "./pdbs"
OUTPUT_CSV = "./preprocess/processed_samples_train.csv"
# =====================================================

def get_knn49_indices(pdb_file_path, target_position):
    """Parse PDB file and find 49 surrounding residues for the target position."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("PDB", pdb_file_path)
        residue_positions = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  
                        residue_positions.append(residue.id[1])
        
        if not residue_positions:
            return None
            
        residue_positions.sort()
        target_idx = residue_positions.index(target_position)
        
        start_idx = max(0, target_idx - 24)
        end_idx = min(len(residue_positions), target_idx + 25)
        return residue_positions[start_idx:end_idx]
    except Exception as e:
        print(f"Failed to parse PDB: {os.path.basename(pdb_file_path)}, Error: {e}")
        return None

def main():
    df = pd.read_csv(INPUT_CSV)
    required_cols = ['UniProtKB_AC-ID', 'position']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df['knn49_index'] = None
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Step 1: Preparing Samples"):
        uniprot_id = str(row['UniProtKB_AC-ID']).strip()
        target_position = int(row['position'])
        
        pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
        pdb_file_path = os.path.join(PDB_FOLDER, pdb_filename)
        
        if not os.path.exists(pdb_file_path):
            df.at[i, 'knn49_index'] = "[]"
            continue
            
        knn_indices = get_knn49_indices(pdb_file_path, target_position)
        df.at[i, 'knn49_index'] = str(knn_indices if knn_indices is not None else [])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Step 1 complete! Sample indices saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
