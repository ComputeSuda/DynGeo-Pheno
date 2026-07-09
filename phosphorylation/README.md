# DynGeo: Phosphorylation Prediction Module
This directory contains the complete implementation of the **phosphorylation prediction** branch of DynGeo.
To capture both the local residue microenvironment and the global protein structural context, DynGeo employs a **Dual-Branch Architecture**. It integrates a local 49-residue subgraph (KNN49) with a global 300-residue full-length graph (P1_300) using protein language model embeddings, structural dynamics, and residue interaction graphs.
---
## Directory Structure
```
phosphorylation/
│
├── train.py                      # Model training and 10-fold CV
├── test.py                       # Ensemble prediction and evaluation
│
├── model/
│   ├── __init__.py
│   └── model.py                  # Dual-branch GCN-GAT-Gate architecture
│
├── preprocess/
│   ├── step1_prepare_samples.py  # Generate local KNN49 indices
│   ├── step2_extract_esm.py      # Full-protein ESM-2 embeddings
│   ├── step3_extract_anm.py      # Full-protein ANM dynamic features
│   ├── step4_construct_graphs.py # Full-protein residue contact graphs
│   └── step5_build_dataset.py    # Split features into KNN & P1 branches
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
1. **Phosphorylation annotation table** (`.csv`)
2. **AlphaFold protein structures** (`.pdb`)
Example files are provided in `example/`. Protein structures should be downloaded according to `pdbs/download_pdbs.txt` and placed inside `pdbs/`.
### Required CSV Format
The phosphorylation table must contain at least the following columns:
| Column | Description |
| :--- | :--- |
| `UniProtKB_AC-ID` | UniProt accession |
| `position` | Phosphorylation site residue position (1-based) |
| `Regulatory_Disease` | Binary label (e.g., 1 for regulatory/disease site, 0 for non-regulatory) |
> **Note:** For inference on independent test datasets, only the site information (`UniProtKB_AC-ID` and `position`) is required. The label column is used only for model evaluation.
---
## 2. Preprocessing Pipeline
The preprocessing workflow converts raw site records and AlphaFold structures into model-ready tensors. 
All scripts assume execution from the `phosphorylation/` directory.
> **Efficiency Note:** All computationally expensive features (ESM embeddings, ANM dynamics and structural graphs) are extracted once at the full-protein level and cached on disk. Sample-specific tensors are generated only during Step 5 by indexing local and global residue subsets.
### Step 1: Prepare Samples
```bash
python preprocess/step1_prepare_samples.py
```
- **Input:** `example/train.csv`
- **Output:** `preprocess/processed_samples_train.csv`
- **Action:** Identifies the local 49-residue neighborhood (KNN49) surrounding each phosphorylation site.
### Step 2: Extract ESM Embeddings
```bash
python preprocess/step2_extract_esm.py
```
- **Input:** `preprocess/processed_samples_train.csv` & AlphaFold structures
- **Output:** `features/esm_128d/*.npy`
- **Action:** Encodes full protein sequences using the pretrained ESM-2 model. The original 1280-dim representations are projected to 128-dim using the provided PCA parameters (`features/pca_params.npy`).
### Step 3: Extract ANM Dynamics
```bash
python preprocess/step3_extract_anm.py
```
- **Input:** `preprocess/processed_samples_train.csv`
- **Output:** `features/anm_front_90d/*.npy` and `features/anm_last_90d/*.npy`
- **Action:** Extracts **full-protein** residue dynamic representations via Anisotropic Network Model (ANM) normal mode analysis. Generates two complementary feature sets: low-frequency (global motion) and high-frequency (local motion).
### Step 4: Construct Structural Graphs
```bash
python preprocess/step4_construct_graphs.py
```
- **Input:** `preprocess/processed_samples_train.csv`
- **Output:** `features/contact_maps/*.npz`
- **Action:** Generates **full-protein** residue adjacency matrices based on C-alpha distance cutoffs from AlphaFold structures.
### Step 5: Build Dataset (Branch Splitting)
```bash
python preprocess/step5_build_dataset.py
```
- **Input:** `preprocess/processed_samples_train.csv` + All extracted full-protein features
- **Output:** `dataset/train_dataset.npz`
- **Action:** For each phosphorylation site, Step 5 converts the cached full-protein representations into two complementary branches:
  - **Local branch:** Residues indexed by KNN49.
  - **Global branch:** Full protein padded/cropped to 300 residues (providing background context).
  No feature extraction is performed during this step.
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
- Parallel processing of both KNN49 and P1_300 branches
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
**🔗 Download Link:** [Baidu Netdisk](https://pan.baidu.com/s/1hJ1Y6XPR4teGLPYMkUr7-Q?1ght) (Extraction code: `wkev`)
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
> **Note:** The preprocessing scripts are demonstrated using the training dataset (`train.csv`) as an example. For an independent test dataset, the identical preprocessing pipeline should be applied after replacing the input CSV with the corresponding test site file. No modifications to the feature extraction algorithms are required.
Because all preprocessing scripts operate directly on AlphaFold structures and site annotations, no retraining or feature engineering is required when applying DynGeo to new phosphorylation datasets.
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
Raw phosphorylation site table
        │
        ▼
Step 1
Generate KNN49 residue indices
        │
        ▼
Step 2
Full-protein ESM embeddings
        │
        ▼
Step 3
Full-protein ANM dynamics
        │
        ▼
Step 4
Full-protein structural graph
        │
        ▼
Step 5
Generate
 ├── Local KNN49 branch
 └── Global P1_300 branch
        │
        ▼
train.py
        │
        ▼
10-fold models
        │
        ▼
test.py
        │
        ▼
Ensemble prediction
```
---
## 7. Model Inputs
DynGeo receives two complementary structural branches for each phosphorylation site. These tensors are automatically loaded by `train.py` and `test.py`.
**Local Branch (KNN49)**
- ESM embedding:      `(49, 128)`
- ANM low-frequency:  `(49, 90)`
- ANM high-frequency: `(49, 90)`
- Contact graph:      `(49, 49)`
**Global Branch (P1_300)**
- ESM embedding:      `(300, 128)`
- ANM low-frequency:  `(300, 90)`
- ANM high-frequency: `(300, 90)`
- Contact graph:      `(300, 300)`
