# DynGeo-Pheno

**DynGeo-Pheno** is a geometric deep learning framework that integrates evolutionary sequence representations (**ESM-2**) with multi-band structural dynamics derived from Elastic Network Models (**ANM / GNM**) through edge-biased Graph Attention Networks (**GATv2**).

It provides two task-specific pipelines, both evaluated under strict **protein-disjoint** protocols (zero UniProt overlap between training and validation folds):

1. **Disease-Associated Phosphorylation Site Prediction** — full-length protein contact graph architecture (`PhosDualStreamGNN`), contacts within 10 Å, macro (whole-protein) vs. micro (8 Å site mask) dual-stream readout.
2. **Pathogenic Missense Mutation Prediction** — local 3D ego-subgraph architecture (`MutEgoGATBatch`), k=32 neighbors within 12 Å, center vs. micro-environment dual-stream readout.

**Core design**: sequence and dynamics features are fused via a **residual gated mechanism**, and low-/high-frequency dynamics are combined by a learnable **FrequencyBandGating** module. The same network architecture supports all ablation presets — feature masking is handled entirely at the data-assembly level.

---

## Repository Structure

```text
DynGeo-Pheno/
├── features/
│   └── feature_plugins.py                     # Derive RMSF / CC / PRS / Stiffness
│                                              # from ANM-GNM spectral caches (multiprocessing)
├── phosphorylation/
│   ├── model/
│   │   └── phos_gat_dual_stream.py            # PhosDualStreamGNN model definition
│   ├── preprocess/
│   │   └── build_phos_graphs.py               # Build full-length protein graphs (Library A & B)
│   ├── train_cv.py                            # 10-fold cross-validation training script
│   └── utils/
│       └── dyn_stepladder_selector.py         # Ablation preset & frequency-band tensor assembler
├── mutation/
│   ├── model/
│   │   └── mut_gat_ego.py                     # MutEgoGATBatch model definition
│   ├── preprocess/
│   │   ├── build_mutation_libraryA_graphs.py  # Build 10-fold CV ego-graphs (Library A)
│   │   └── build_mutation_libraryB_graphs.py  # Build external test ego-graphs (DDD & Cancer Hotspot)
│   ├── train_mutation_cv.py                   # 10-fold cross-validation training script
│   └── utils/
│       └── mut_stepladder_selector.py         # Ablation preset & frequency-band tensor assembler
└── README.md
```

---

## Environment Setup

```bash
conda create -n dyngeo python=3.10 -y
conda activate dyngeo

# PyTorch and PyTorch Geometric (adjust CUDA version for your system)
pip install torch torchvision torchaudio
pip install torch_geometric

# Scientific dependencies
pip install numpy scipy pandas scikit-learn tqdm openpyxl prody
```

---

## Data Preparation

Place precomputed inputs under `data/` at the repository root:

```text
data/
├── AlphaFold_pdb/            # AlphaFold structures used for graph construction
├── Raw_Spectral_Cache/       # ANM/GNM eigendecomposition caches: {stem}_spectral.npz
│                             #   keys: coords, resnums, anm_eigvals, anm_u,
│                             #         gnm_eigvals, gnm_vecs
├── ESM_128d/                 # 128-d PCA-projected wild-type ESM-2 embeddings
└── ESM_128d_mut/             # 128-d PCA-projected mutant ESM-2 embeddings
```

Benchmark tables (standardized to the columns `ACC_ID`, `pdb_stem`, `site_str`, `site`, `plddt`, `Label`, `fold`) should be placed at:

```text
phosphorylation/processed_graphs/data_scheme1/   # Library A / Library B tables
mutation/processed_graphs/                       # Library A / B1 / B2 tables
```

### Step 1 — Derive Dynamics Features

From the spectral caches, derive all physical descriptors:

```bash
python features/feature_plugins.py
```

Outputs (written to `data/Extracted_Features/`):

| Directory      | Contents                                                             |
| -------------- | -------------------------------------------------------------------- |
| `RMSF/`      | ANM & GNM root-mean-square fluctuations across all frequency slices  |
| `CC/`        | Cross-correlation matrices (contact-masked, normalized)              |
| `PRS/`       | Perturbation-response scanning: sensitivity & effectiveness per band |
| `Stiffness/` | Analytic mechanical stiffness matrix                                 |

Frequency slices generated per protein: `first10/20/30/40/60`, `last10/20/30/40/60`, `first2pct`, `last2pct`, `all` (2% = length-adaptive).

---

## Task 1: Disease-Associated Phosphorylation Site Prediction

### 1. Build Full-Protein Graph Caches

```bash
python phosphorylation/preprocess/build_phos_graphs.py
```

Constructs full-length contact graphs (≤10 Å) with per-site 8 Å micro-environment masks, supporting multiple candidate sites per protein in a single forward pass.

### 2. 10-Fold Cross-Validation Training

```bash
python phosphorylation/train_cv.py \
    --exp_preset exp_3c_mix_30 \
    --num_layers 2 \
    --hidden_dim 256 \
    --lr 1e-4 \
    --dropout 0.2 \
    --epochs 20
```

---

## Task 2: Pathogenic Missense Mutation Prediction

### 1. Build Ego-Graph Caches

```bash
python mutation/preprocess/build_mutation_libraryA_graphs.py
python mutation/preprocess/build_mutation_libraryB_graphs.py
```

Each mutation site becomes a local ego-graph  carrying the ESM difference `Δh = h_WT − h_MUT`, all-band dynamics node features, and continuous mechanics-guided edge attributes (distance, inverse distance, CC, stiffness).

### 2. 10-Fold Cross-Validation Training

```bash
python mutation/train_mutation_cv.py \
    --exp_preset exp_3c_mix_30 \
    --num_layers 1 \
    --hidden_dim 128 \
    --lr 1e-4 \
    --dropout 0.3 \
    --batch_size 256 \
    --epochs 20
```

---

## Ablation Presets

Both training scripts accept `--exp_preset` for on-the-fly feature slicing. All presets share the identical network architecture — ablation is realized purely through input masking.

| Preset                | Node Features               | Purpose                                       |
| --------------------- | --------------------------- | --------------------------------------------- |
| `exp_0_seq_only`    | —                          | ESM-only lower bound                          |
| `exp_dyn_only`      | low30 + high30 (ESM masked) | Dynamics-only                                 |
| `exp_1a_rmsf_only`  | RMSF (ANM+GNM, first30)     | Flexibility mechanism                         |
| `exp_1b_prs_only`   | PRS (Sens+Eff, first30)     | Allosteric perturbation mechanism             |
| `exp_1c_cc_only`    | — (CC edge bias)           | Correlated-motion mechanism                   |
| `exp_1d_stiff_only` | — (stiffness edge)         | Elastic stiffness mechanism                   |
| `exp_2a`–`2g`    | first10 → 60, 2%, all      | Low-frequency modal-capacity scan             |
| `exp_2h`–`2m`    | last10 → 60, 2%            | High-frequency negative controls              |
| `exp_3a`–`3f`    | low + high dual-band        | Adaptive band gating (**3c = primary**) |


