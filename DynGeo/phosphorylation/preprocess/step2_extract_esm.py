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
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        three2one = {
            'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
            'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
            'THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'
        }
        seq = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] == " ":
                        seq.append(three2one.get(res.get_resname().strip(), 'X'))
        return "".join(seq)

    def extract_protein(self, protein_id, pdb_path):
        out_path = os.path.join(OUTPUT_FOLDER, f"{protein_id}.npy")
        if os.path.exists(out_path):
            return np.load(out_path)

        seq = self.get_sequence_from_pdb(pdb_path)
        if not seq:
            return None

        # Sliding window extraction for 1280-dim features
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
            feat_1280[s:e] = out["representations"][CONFIG["repr_layer"]][0, 1:e-s+1].cpu().numpy()
            del tokens, out
            if CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # PCA dimensionality reduction
        feat_128 = self.pca.transform(feat_1280).astype(np.float32)
        np.save(out_path, feat_128)
        return feat_128

def main():
    df = pd.read_csv(INPUT_CSV)
    extractor = ESMExtractor()
    
    unique_proteins = df['UniProtKB_AC-ID'].unique()
    for pid in tqdm(unique_proteins, desc="Step 2: Extracting ESM Features"):
        pdb_path = os.path.join(PDB_FOLDER, f"AF-{pid}-F1-model_v4.pdb")
        if os.path.exists(pdb_path):
            extractor.extract_protein(pid, pdb_path)
            
    print("Step 2 complete! ESM 128-dim features extracted.")

if __name__ == "__main__":
    main()
