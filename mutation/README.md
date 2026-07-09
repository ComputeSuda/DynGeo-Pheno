# DynGeo: Mutation Prediction Module
This directory contains the complete implementation of the **mutation prediction** branch of DynGeo.
DynGeo integrates protein language model embeddings, structural dynamics, and residue interaction graphs to predict the functional effects of missense mutations.
---
## Directory Structure
```
mutation/
│
├── train.py                      # Model training and 10-fold CV
├── test.py                       # Ensemble prediction and evaluation
│
├── model/
│   ├── __init__.py
│   └── model.py                  # GCN-GAT-Gate architecture
│
├── preprocess/
│   ├── step1_prepare_samples.py  # Generate residue neighborhoods
│   ├── step2_extract_esm.py      # Mutation-specific ESM-2 embeddings
│   ├── step3_extract_anm.py      # Protein-level ANM dynamic features
│   ├── step4_construct_graphs.py # Protein-level residue contact graphs
│   └── step5_build_dataset.py    # Pack tensors into .npz
│
├── utils/
│   ├── data.py                   # Data loading and splitting
│   ├── loss.py                   # Focal loss
│   └── metrics.py                # Evaluation metrics
│
├── example/                      # Example CSV files
├── features/                     # Precomputed PCA params & feature cache
├── pdbs/                         # AlphaFold structures
├── checkpoints/                  # Downloaded from Baidu
│
└── README.md
```
---
## 1. Input Data
The preprocessing pipeline requires two inputs:
1. **Mutation annotation table** (`.csv`)
2. **AlphaFold protein structures** (`.pdb`)
Example files are provided in `example/`. Protein structures should be downloaded according to `pdbs/download_pdbs.txt` and placed inside `pdbs/`.
### Required CSV Format
The mutation table must contain at least the following columns to support mutation-specific embedding extraction:
| Column | Description |
| :--- | :--- |
| `UniProtKB_AC-ID` | UniProt accession |
| `position` | Residue position (1-based, matching PDB numbering) |
| `wild_aa` | Wild-type amino acid (e.g., 'R') |
| `mut_aa` | Mutant amino acid (e.g., 'S') |
| `clinvar_pathogenicity` | Binary label (training only) |
> **Note:** For inference on independent test datasets, only the mutation information (`UniProtKB_AC-ID`, `position`, `wild_aa`, `mut_aa`) is required. The label column is used only for model evaluation.
---
## 2. Preprocessing Pipeline
The preprocessing workflow converts raw mutation records and AlphaFold structures into model-ready tensors. 
All scripts assume execution from the `mutation/` directory.
### Step 1: Prepare Samples
```bash
python preprocess/step1_prepare_samples.py
```
- **Input:** `example/train.csv`
- **Output:** `preprocess/processed_samples_train.csv`
- **Action:** Identifies the local 49-residue neighborhood (KNN49) surrounding each mutation site.
### Step 2: Extract Mutation-Specific ESM Embeddings
```bash
python preprocess/step2_extract_esm.py
```
- **Input:** `preprocess/processed_samples_train.csv` & AlphaFold structures
- **Output:** `features/esm_128d/*.npy`
- **Action:** Generates mutation-specific embeddings using the pretrained ESM-2 model. 
  - For each protein, the Wild-Type (WT) sequence is encoded once and cached (`{pid}_WT.npy`) for efficiency.
  - For each mutation, the Mutant (Mut) sequence is encoded, and the residue-level difference (`WT - Mut`) is computed.
  - The original 1280-dim representations are projected to 128-dim using the provided PCA parameters (`features/pca_params.npy`).
  - The final output for a mutation (e.g., R368S in protein P49711) is saved as `P49711_R368S.npy`.
### Step 3: Extract ANM Dynamics
```bash
python preprocess/step3_extract_anm.py
```
- **Input:** `preprocess/processed_samples_train.csv`
- **Output:** `features/anm_front_90d/*.npy` and `features/anm_last_90d/*.npy`
- **Action:** Extracts **protein-level** residue dynamic representations via Anisotropic Network Model (ANM) normal mode analysis based on the WT structure. Generates two complementary feature sets: low-frequency (global motion) and high-frequency (local motion).
### Step 4: Construct Structural Graphs
```bash
python preprocess/step4_construct_graphs.py
```
- **Input:** `preprocess/processed_samples_train.csv`
- **Output:** `features/contact_maps/*.npz`
- **Action:** Generates **protein-level** residue adjacency matrices based on C-alpha distance cutoffs from WT AlphaFold structures.
### Step 5: Build Dataset
```bash
python preprocess/step5_build_dataset.py
```
- **Input:** `preprocess/processed_samples_train.csv` + All extracted features
- **Output:** `dataset/train_dataset.npz`
- **Action:** Assembles features into a single compressed dataset. It loads the mutation-specific ESM embeddings and the protein-level structural features (ANM, Contact Maps), extracts the 49-residue subgraph for each, and pads/crops them to standardized shapes. This file is directly consumed by `train.py`.
---
## 3. Training
After preprocessing, initiate model training:
```bash
python train.py
```
The training script performs:
- 10-fold stratified cross-validation
- Focal loss optimization
- Automatic early stopping & learning-rate scheduling
**Outputs:**
- `checkpoints/model_0.keras` to `model_9.keras` (10-fold models)
- `cross_val_results.xlsx` (Per-fold metrics)
- `checkpoints/train_summary.json` (Optimal threshold & OOF metrics)
The optimal decision threshold estimated from cross-validation is stored in `train_summary.json` and will be reused automatically during testing.
---
## 4. Testing & Inference
We provide pre-trained 10-fold models (`model_fold_1.keras` to `model_fold_10.keras`) for direct inference without retraining.
### 4.1 Download Pre-trained Models
Due to their large size, the model files are hosted on Baidu Netdisk. **You must download them before testing.**
**🔗 Download Link:** [Baidu Netdisk](https://pan.baidu.com/s/109AK8jtM2114dd9-KZEU1w?pwd=wkev) (Extraction code: `wkev`)
**Setup:**
1. Download the `checkpoints` folder from the link above.
2. Place it in `mutation/checkpoints/` (should contain `model_fold_1.keras` to `model_fold_10.keras` and `train_summary.json`).
3. Alternatively, you can find the link in the `model.txt` file in the repository root.
### 4.2 Run Inference on Test Set
Ensure your test dataset is prepared as `dataset/test_dataset.npz` (see Section 5). Then run:
```bash
python test.py
```
The script will:
- Load all 10 pre-trained models
- Perform ensemble prediction
- Apply the optimal threshold from `train_summary.json`
- Output metrics to `test_results.xlsx`
### 4.3 Predicting Your Own Data
To predict mutation effects on your own proteins:
1. Prepare your CSV with `UniProtKB_AC-ID`, `position`, `wild_aa`, `mut_aa`
2. Download corresponding AlphaFold PDB files into `pdbs/`
3. Run Step 1 to Step 5 to generate `dataset/test_dataset.npz`
4. Run `python test.py`

---
## 5. Generalization to Independent Datasets
> **Note:** The preprocessing scripts are demonstrated using the training dataset (`train.csv`) as an example. For an independent test dataset, the identical preprocessing pipeline should be applied after replacing the input CSV with the corresponding test mutation file. No modifications to the feature extraction algorithms are required.
To generate features for an independent test dataset, simply replace the input CSV in Step 1 to Step 5:
```text
example/train.csv  →  example/test.csv
```
Execute the pipeline unchanged:
```text
Step 1 → Step 2 → Step 3 → Step 4 → Step 5
```
The resulting tensors will be packaged as `dataset/test_dataset.npz`, which is directly accepted by `python test.py`.
---
## 6. Workflow Summary
```text
Mutation CSV (train/test)
        │
        ▼
Step 1: Prepare KNN49 residue neighborhoods
        │
        ▼
Step 2: Mutation-specific ESM-2 embeddings (WT - Mut, 128-d)
        │
        ▼
Step 3: Protein-level ANM dynamics (90-d)
        │
        ▼
Step 4: Protein-level Contact graphs
        │
        ▼
Step 5: Extract subgraphs & Build dataset.npz
        │
        ▼
train.py  →  10-fold models  →  test.py  →  Ensemble prediction
```

