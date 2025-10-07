# ST2HE: Virtual Histology from High-Resolution Spatial Transcriptomics

**ST2HE** is a cross-platform, generative framework that synthesizes **virtual Hematoxylin & Eosin (H&E)** histology images directly from **high-resolution spatial transcriptomics (HR-ST)** data.  
By integrating DAPI morphology and spatial transcript coordinates using a **one-step diffusion model**, ST2HE bridges the gap between molecular and histological domains—enabling interpretable, scalable annotation of HR-ST datasets.

---

## Key Features

- **Virtual H&E generation** directly from HR-ST data (e.g. Xenium, NanoString CosMx, MERFISH).  
- **One-step diffusion model** (Pix2Pix-Turbo backbone) for fast and high-fidelity image translation.  
- **Conditional & unconditional modes:**
  - `ST2HE-CondGen` — tissue-specific generation using CLIP text prompts.  
  - `ST2HE-UncondGen` — generalizable model for unseen tissues.  
- **Cross-platform generalization** demonstrated on:
  - Xenium Breast Cancer
  - NanoString NSCLC
  - Xenium Kaposi’s Sarcoma
- **Multi-colored transcript visualization** to maximize morphology-gene expression coupling.
- Compatible with pathology foundation models (e.g., **CONCH**) for downstream classification and annotation.

---

## Conceptual Overview

The ST2HE framework converts high-resolution spatial transcriptomics (HR-ST) data into virtual H&E images through a one-step diffusion model built on Pix2Pix-Turbo, integrating DAPI morphology and transcript spatial coordinates.

<p align="center">
  <img src="figures/fig1_overview.png" alt="ST2HE framework overview" width="700"/>
</p>

**Figure 1. Overview of ST2HE Framework.**
- **(a)** ST2HE bridges transcriptomic data with histological features, enabling automated annotation of tumor subtypes.  
- **(b)** Architecture: Encoder–U-Net–Decoder backbone with CLIP-based text conditioning (“This is a {tissue} H&E image” or “dapi2he”).  
- **(c)** Input preparation: DAPI + transcript overlay → embedded input.  
- **(d–e)** Dual modes:  
  - **ST2HE-CondGen:** tissue-conditioned generation (e.g., lung input → lung virtual H&E).  
  - **ST2HE-UncondGen:** tissue-independent generalization for unseen tissues (e.g., skin input → skin virtual H&E).

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/Huang-AI4Medicine-Lab/ST2HE.git
cd ST2HE
```

### 2. Create environment
```bash
conda env create -f environment.yml
conda activate st2he
```

*(or use pip with `requirements.txt`)*

### 3. Optional: Enable GPU acceleration
Ensure you have **CUDA ≥ 11.8** and **PyTorch ≥ 2.1** installed.

---

## Quickstart: Generate Virtual H&E

### Example: Xenium sample
```bash
python src/st2he/infer.py     --input data/xenium/sample1_dapi.png     --coords data/xenium/sample1_coords.csv     --mode condgen     --prompt "This is a breast H&E image"     --output results/sample1_virtual_he.png
```

### Example: Unseen tissue (Kaposi’s Sarcoma)
```bash
python src/st2he/infer.py     --input data/ks/core_01_dapi.png     --coords data/ks/core_01_coords.csv     --mode uncongen     --prompt "dapi2he"
```

---

## Citation

If you use this repository, please cite:

```bibtex
@article{liu2025st2he,
  title={ST2HE: A Cross-Platform Framework for Virtual Histology and Annotation of High-Resolution Spatial Transcriptomics Data},
  author={Liu, Zhentao and Das, Arun and Meng, Wen and Chiu, Yu-Chiao and Gao, Shou-Jiang and Huang, Yufei},
  year={2025},
  journal={Preprint},
  note={University of Pittsburgh & UPMC Hillman Cancer Center}
}
```

---

## Acknowledgements

Supported by NIH (U01CA279618, R21GM155774, CA096512, CA284554, CA278812, CA291244, CA124332, R35GM154967) and UPMC Hillman Cancer Center Startup Funds.  
Computing resources provided by **University of Pittsburgh Center for Research Computing (HTC Cluster, S10OD028483)**.

---

**Maintainer:** Huang AI4Medicine Lab  
University of Pittsburgh | UPMC Hillman Cancer Center  
yuh119@pitt.edu
