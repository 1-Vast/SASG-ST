# Dataset Download Sources

This repository does not commit raw datasets. Download the data from the original public sources listed below.

## DLPFC

- Source: spatialLIBD, LIBD Human DLPFC 10x Genomics Visium data.
- URL: https://research.libd.org/spatialLIBD/
- Interactive/raw-data portal: https://spatial.libd.org/spatialLIBD/
- Use sample `151675` for the example in the repository [README](../README.md).
- Required annotation for the reproduction command: `sce.layer_guess`.

## Mouse Brain Serial Sections

- Source: 10x Genomics Visium mouse brain serial section datasets.
- Section 1, Sagittal-Anterior: https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-0-0
- Section 1, Sagittal-Posterior: https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-posterior-1-standard-1-0-0
- Section 2, Sagittal-Anterior: https://www.10xgenomics.com/datasets/mouse-brain-serial-section-2-sagittal-anterior-1-standard-1-1-0
- Section 2, Sagittal-Posterior: https://www.10xgenomics.com/datasets/mouse-brain-serial-section-2-sagittal-posterior-1-standard-1-1-0

## Coronal Mouse Brain

- Source: 10x Genomics Visium adult mouse brain coronal dataset.
- URL: https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-0-0

## Human Breast Cancer

- Source: 10x Genomics Visium Human Breast Cancer Block A Section 1.
- Study page: https://singlecell.broadinstitute.org/single_cell/study/SCP1256/visium-demo-study
- Filtered matrix: https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5
- Spatial files: https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz

## Mouse Embryo

- Source: MOSTA, Mouse Organogenesis Spatiotemporal Transcriptomic Atlas.
- URL: https://db.cngb.org/stomics/mosta/download/
- Download the corresponding `.MOSTA.h5ad` files, such as `E11.5_E1S1.MOSTA.h5ad`, `E12.5_E1S2.MOSTA.h5ad`, `E13.5_E1S4.MOSTA.h5ad`, and `E14.5_E1S3.MOSTA.h5ad`.
