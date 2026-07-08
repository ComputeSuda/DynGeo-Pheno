# DynGeo: Dynamics-Integrated Geometric Deep Learning for Protein Functional Site Prediction

DynGeo is a unified geometric deep learning framework for modeling protein functional perturbations induced by diverse molecular events. By integrating protein language representations, structural dynamics, and residue-level geometric topology, DynGeo learns a shared representation of protein functional landscapes while allowing task-specific adaptations for different biological problems, including pathogenic missense mutations and regulatory phosphorylation sites.

Without relying on any prior functional annotations during training, DynGeo provides robust predictive power and independent mechanistic insights into how molecular variations alter protein function.

---

## 🌟 Key Features

- **Unified Multi-Scale Framework**: A consistent geometric learning paradigm that jointly models sequence, structural dynamics, and spatial topology.
- **Flexible Task Adaptation**: DynGeo maintains a unified learning framework while supporting different biological tasks through task-specific evolutionary representations and graph construction strategies.
- **Structural Dynamics & ADF**: Incorporates Anisotropic Network Model (ANM) normal mode analysis to capture multiscale protein flexibility. Features are integrated via a novel **Adaptive Dynamic Fusion (ADF)** module that uses feature-wise gating.
- **Geometry-Aware Graph Learning**: Protein structures are represented as residue contact graphs and processed by a parallel Modified GCN and Two-stage GAT hybrid module to learn local and global structural descriptors.

---

## 🏗️ Framework Overview

DynGeo follows a unified feature extraction and graph learning framework. The overall workflow is shared across different prediction tasks, while only the task-specific evolutionary representation and graph construction strategy are adapted according to the biological problem.

```text
        Protein sequence / structure
                         │
                         ▼
        Protein Language Model (ESM-2)
                         │
                         ▼
        Structural Dynamics (ANM)
                         │
                         ▼
      Residue Contact Graph Construction
                         │
                         ▼
         Adaptive Dynamic Fusion (ADF)
                         │
                         ▼
       Hybrid Graph Learning (GCN + GAT)
                         │
                         ▼
            Task-specific Prediction
```

> **Note**: The neural network architecture remains identical across prediction tasks. Task-specific adaptations are strictly limited to the evolutionary representation (e.g., WT−Mut residual embeddings for missense mutations) and the graph construction strategy (e.g., mutation-centered local graphs vs. dual-scale phosphorylation graphs), while all subsequent learning modules are shared.

---

## 📁 Repository Structure

The repository contains two task-specific implementations built upon the same DynGeo framework. Both modules share the overall design philosophy while differing only in task-specific preprocessing and graph construction.

```text
DynGeo/
│
├── mutation/                     # Implementation for pathogenic missense mutations
├── phosphorylation/              # Implementation for regulatory phosphorylation sites
├── environment.yml               # Shared Conda environment configuration
└── README.md                     # Current file
```

---

## 🚀 Quick Start

### 1. Environment Setup

We recommend using `conda` to manage the environment. DynGeo requires Python ≥3.9, PyTorch ≥2.5, and TensorFlow ≥2.20.

```bash
git clone https://github.com/ComputeSuda/DynGeo.git
cd DynGeo

# Create the environment using the provided yml file
conda env create -f environment.yml
conda activate dl_env
```

### 2. Download AlphaFold Structures

Both tasks require high-quality structural inputs. Please refer to the `pdbs/download_pdbs.txt` file in each task directory to fetch the corresponding PDB files from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/).

### 3. Run a Task

Choose the biological task you want to model and navigate to its dedicated directory. Each directory contains a detailed, step-by-step README for reproducing the pipeline.

**For Phosphorylation Site Prediction:**
```bash
cd phosphorylation
# Follow instructions in phosphorylation/README.md
```

**For Mutation Effect Prediction:**
```bash
cd mutation
# Follow instructions in mutation/README.md
```

---

## 🔄 Workflow Paradigm

DynGeo's data processing follows a highly efficient **"Extract-Once, Index-Twice"** paradigm. Computationally expensive physics and language features are extracted once per full-length protein and cached, preventing computational redundancy.

```text
       Raw protein annotations (Mutations or PTMs)
                         │
                         ▼
                Sample preparation
                         │
                         ▼
         Protein-level feature extraction
                         │
                         ├── ESM-2 embeddings
                         ├── ANM multi-scale dynamics
                         └── Contact graphs
                         │
                         ▼
         Task-specific graph construction
                         │
                         ▼
             Adaptive Dynamic Fusion
                         │
                         ▼
              Hybrid Graph Learning
                         │
                         ▼
                 Final Prediction
```

---

## 📄 Citation

Citation information will be updated upon publication. If you find our unified framework or mechanistic interpretations useful, please consider starring this repository.

---

## 📬 Contact

If you have any questions, encounter issues, or wish to collaborate, please open an issue in the GitHub repository or contact [20254257025@stu.suda.edu.cn].

***

