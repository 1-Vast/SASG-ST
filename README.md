# SASG-ST

SASG-ST is the code repository for the accompanying paper. It provides the model implementation and reproduction commands for spatial transcriptomics representation learning and spatial domain clustering. Raw datasets and generated outputs are not stored in Git.

## Overview

SASG-ST learns spot-level representations from gene expression and spatial neighborhood graphs. The workflow first builds a spatial graph, trains a graph representation model with augmentation and reconstruction objectives, then clusters the learned embeddings for downstream spatial domain analysis.

![SASG-ST model overview](overview.png)

## Installation

```bash
conda env create -f environment.yml
conda activate sasg-st
```

The environment installs PyTorch 2.5.0 with CUDA 12.1 and Scanpy dependencies.
For mclust clustering, install R and the optional dependencies:

```bash
Rscript -e 'install.packages("mclust", repos="https://cloud.r-project.org")'
python -m pip install rpy2==3.5.17
```

## Dataset Downloads

Download datasets from the original public sources:

- DLPFC: https://research.libd.org/spatialLIBD/
- Mouse Brain Serial Sections: https://www.10xgenomics.com/resources/datasets
- Coronal Mouse Brain: https://www.10xgenomics.com/resources/datasets
- Human Breast Cancer: https://singlecell.broadinstitute.org/single_cell/study/SCP1256/visium-demo-study
- Mouse Embryo MOSTA: https://db.cngb.org/stomics/mosta/download/

More detailed dataset links are listed in [dataset/README.md](dataset/README.md).

## Usage

The example below uses DLPFC section `151675`, saved as `dataset/DLPFC/151675.h5ad`.

### Training

```bash
python main.py \
  --h5 dataset/DLPFC/151675.h5ad \
  --graph_model KNN \
  --k 20 \
  --alpha 0.5 \
  --K 2 \
  --dim 64 \
  --hidden 512 \
  --epochs 1200 \
  --lr 0.0001 \
  --lambda_recon 2.0 \
  --mask_ratio_feat 0.2 \
  --layer_aware \
  --layer_gamma 2.0 \
  --neg_layer_margin 2 \
  --neg_hard_ratio 0.3 \
  --pseudo_layer_bins 7 \
  --normal_aware \
  --normal_margin 1.0 \
  --seed 42
```

### Clustering and evaluation

```bash
python -m model.cluster \
  --npz dataset/DLPFC/151675.augK2_d64_for_cluster.npz \
  --method mclust \
  --n_clusters 7 \
  --pca_dim 15 \
  --h5 dataset/DLPFC/151675.h5ad \
  --label_key sce.layer_guess \
  --calc_acc
```
