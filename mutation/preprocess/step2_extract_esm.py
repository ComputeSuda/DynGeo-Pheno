#!/usr/bin/env python3
import os
import torch
import esm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser
from tqdm import tqdm
import gc

# ======================= Config =======================
INPUT_CSV = "./preprocess/processed_samples_train.csv"
PDB_FOLDER = "./pdbs"
OUTPUT_FOLDER = "./features/esm_128d"
PCA_MODEL_PATH = "./features/pca_params.npy"
LOG_FILE = "./features/esm_extraction.log"

CONFIG = {
    "model_name": "esm2_t33_650M_UR50D",
    "repr_layer": 33,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.float16,
    "window_size": 256,
    "overlap_size": 64,
    "pca_n_components": 128
}
# =====================================================

class ESMExtractor:
    def __init__(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(CONFIG["model_name"])
        self.model = self.model.to(CONFIG["device"], dtype=CONFIG["dtype"])
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        # PCA initialization
        self.pca = None
        if os.path.exists(PCA_MODEL_PATH):
            params = np.load(PCA_MODEL_PATH, allow_pickle=True).item()
            self.pca = PCA(n_components=CONFIG["pca_n_components"])
            self.pca.components_ = params["components"]
            self.pca.mean_ = params["mean"]
            self.pca.explained_variance_ = params["explained_variance"]
            print("Loaded pretrained PCA model.")
        else:
            raise FileNotFoundError(
                f"PCA model file {PCA_MODEL_PATH} not found! "
                f"Please run `python preprocess/fit_pca.py` first."
            )

    def get_sequence_from_pdb(self, pdb_path):
        """Extract amino acid sequence and residue numbers from PDB."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        three2one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'MSE': 'M'
        }
        seq_chars = []
        res_nums = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] == " ":
                        rn = res.get_resname().strip()
                        seq_chars.append(three2one.get(rn, 'X'))
                        res_nums.append(res.id[1])
        return "".join(seq_chars), res_nums

    def _compute_esm_1280(self, seq):
        """Compute raw 1280-dim ESM embeddings via sliding window."""
        feat_1280 = np.zeros((len(seq), 1280), dtype=np.float32)
        starts = []
        start = 0
        while start < len(seq):
            end = min(start + CONFIG["window_size"], len(seq))
            starts.append((start, end))
            if end == len(seq):
                break
            start = end - CONFIG["overlap_size"]

        for s, e in starts:
            data = [("seq", seq[s:e])]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(CONFIG["device"])
            with torch.no_grad():
                out = self.model(tokens, repr_layers=[CONFIG["repr_layer"]], return_contacts=False)
            feat_1280[s:e] = out["representations"][CONFIG["repr_layer"]][0, 1:e - s + 1].cpu().numpy()
            del tokens, out
            if CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        return feat_1280

    def get_wt_embedding(self, protein_id, pdb_path):
        """Get or compute WT ESM embedding (cached as {pid}_WT.npy)."""
        wt_path = os.path.join(OUTPUT_FOLDER, f"{protein_id}_WT.npy")
        if os.path.exists(wt_path):
            return np.load(wt_path)

        seq, res_nums = self.get_sequence_from_pdb(pdb_path)
        if not seq:
            return None

        feat_1280 = self._compute_esm_1280(seq)
        feat_128 = self.pca.transform(feat_1280).astype(np.float32)
        np.save(wt_path, feat_128)
        return feat_128

    def get_mut_embedding(self, mut_seq):
        """Compute Mut ESM embedding (not cached, mutation-specific)."""
        feat_1280 = self._compute_esm_1280(mut_seq)
        feat_128 = self.pca.transform(feat_1280).astype(np.float32)
        return feat_128

    def extract_mutation(self, protein_id, pdb_path, position, wt_aa, mut_aa):
        """Compute mutation-specific embedding (WT - Mut) and save."""
        # Output filename: {pid}_{wt}{pos}{mut}.npy
        out_name = f"{protein_id}_{wt_aa}{position}{mut_aa}.npy"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        if os.path.exists(out_path):
            return np.load(out_path)

        # Step 1: Get WT embedding (cached per protein)
        wt_embedding = self.get_wt_embedding(protein_id, pdb_path)
        if wt_embedding is None:
            return None

        # Step 2: Build mutated sequence
        seq, res_nums = self.get_sequence_from_pdb(pdb_path)
        if not seq:
            return None

        # Find the 0-based index corresponding to the mutation position
        try:
            seq_idx = res_nums.index(position)
        except ValueError:
            print(f"Position {position} not found in residue list for {protein_id}")
            return None

        # Safety check: verify WT amino acid matches
        if seq[seq_idx] != wt_aa:
            print(f"Warning: {protein_id} pos {position}, expected {wt_aa} but found {seq[seq_idx]}")

        # Apply mutation
        mut_seq = seq[:seq_idx] + mut_aa + seq[seq_idx + 1:]

        # Step 3: Compute Mut embedding
        mut_embedding = self.get_mut_embedding(mut_seq)

        # Step 4: Compute WT - Mut difference
        diff_embedding = (wt_embedding - mut_embedding).astype(np.float32)
        np.save(out_path, diff_embedding)
        return diff_embedding


def main():
    df = pd.read_csv(INPUT_CSV)

    # Verify required columns
    required_cols = ['UniProtKB_AC-ID', 'position']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Handle column name variants for wild/mut amino acids
    wt_col = 'wild_aa' if 'wild_aa' in df.columns else ('wild_a' if 'wild_a' in df.columns else None)
    mut_col = 'mut_aa' if 'mut_aa' in df.columns else ('mut_a' if 'mut_a' in df.columns else None)
    if wt_col is None or mut_col is None:
        raise ValueError("CSV must contain wild_aa/wild_a and mut_aa/mut_a columns")

    extractor = ESMExtractor()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step 2: Extracting Mutation ESM Features"):
        pid = str(row['UniProtKB_AC-ID']).strip()
        position = int(row['position'])
        wt_aa = str(row[wt_col]).strip().upper()
        mut_aa = str(row[mut_col]).strip().upper()

        pdb_path = os.path.join(PDB_FOLDER, f"AF-{pid}-F1-model_v4.pdb")
        if not os.path.exists(pdb_path):
            print(f"PDB not found for {pid}, skipping")
            continue

        extractor.extract_mutation(pid, pdb_path, position, wt_aa, mut_aa)

    print("Step 2 complete! Mutation-specific ESM features extracted.")

if __name__ == "__main__":
    main()
