# BraTS datasets

KnowNet is trained and evaluated on the BraTS 2021, BraTS 2020 and BraTS 2023
brain-tumor segmentation benchmarks. This directory only holds the data-loading
code; the NIfTI volumes are **not** part of the repository (they are gitignored).

## Download links

| Dataset | Official page | Mirror |
|---------|---------------|------------------------------|
| BraTS 2021 | [RSNA-ASNR-MICCAI BraTS 2021](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) | [Kaggle: BraTS2021](https://www.kaggle.com/datasets/syedsajid/brats2021) · Baidu: _to be filled_ |
| BraTS 2020 | [MICCAI BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) | [Kaggle: BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) · Baidu: _to be filled_ |
| BraTS 2023 | [BraTS 2023 Adult Glioma](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282) | [Kaggle: BraTS2023](https://www.kaggle.com/datasets/nicozoro56/brats2023) · Baidu: _to be filled_ |

> Baidu Netdisk mirrors will be added here (`链接: … 提取码: …`). Until then, use
> the official pages or the Kaggle mirrors above.

## Expected layout

Each release is a flat folder of case directories:

```
data/
├── BraTS2021/
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
├── BraTS2020/
│   └── BraTS2020_00000/ ...      # same naming scheme
└── BraTS2023/
    └── BraTS2023_00000/ ...      # same naming scheme
```

The loader discovers cases with `case_glob` (`BraTS*_*` by default) and requires the
`{case_id}_seg.nii.gz` file next to the four modality files. Set `data.data_root`
in `configs/config.yaml` (default `./data/BraTS2021`).

## Splits

The case-level split is generated deterministically from `data.split_ratio` and
`data.split_seed` (8:1:1 by default):

| Dataset | Total | Train | Val | Test |
|---------|:-----:|:-----:|:---:|:----:|
| BraTS 2021 | 1,251 | 1,000 | 125 | 126 |
| BraTS 2020 | 369 | 295 | 37 | 37 |
| BraTS 2023 | 1,251 | 1,000 | 125 | 126 |

BraTS 2021 and BraTS 2023 share one cohort, so the same partition applies to both
versions; only the annotations differ.

## Pre-processing performed by the code

- Label `4` (enhancing tumor) is remapped to `3` at load time.
- Each sequence is z-score normalized per scan and per channel; no statistics are
  shared across cases.
- Training crops 128³ regions centred on a randomly chosen tumour voxel.
- Validation/test keep the full volume and use sliding-window inference.

No additional skull stripping or resampling is applied: the official BraTS
pre-processed releases are used as provided.

## Disk usage

| Dataset | Approximate size |
|---------|------------------|
| BraTS 2021 | ~12 GB |
| BraTS 2020 | ~4 GB |
| BraTS 2023 | ~12 GB |
