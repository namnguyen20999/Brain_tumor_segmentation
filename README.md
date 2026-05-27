# Brain Tumor Segmentation

U-Net (ResNet34 backbone) segmentation experiments on two public brain tumor MRI datasets, implemented as Google Colab notebooks.

---

## Notebooks

| Notebook | Dataset | Input format |
|---|---|---|
| `brain_tumor_segmentation.ipynb` | Kaggle Brain Tumor Segmentation | PNG images + masks |
| `matlab_brain_segmentation.ipynb` | figshare 1512427 (Cheng et al.) | `.mat` files → converted to PNG |

---

## How to run

Both notebooks are designed to run on **Google Colab** with your data stored in **Google Drive**.

### 1. Open in Colab

Upload the notebook to your Google Drive and open it with Google Colab, or use **File → Open notebook → GitHub** to open it directly.

### 2. Mount your Google Drive

The first cell mounts your Drive automatically:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Approve the prompt when it appears.

### 3. Upload your dataset to Google Drive

**For `brain_tumor_segmentation.ipynb`** — expects a folder with two subfolders:

```
MyDrive/
└── brain_segmentation_dataset/
    └── archive/
        ├── images/   ← brain MRI images (.png)
        └── masks/    ← binary tumor masks (.png, same filenames as images)
```

Dataset source: [Brain Tumor Segmentation on Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

**For `matlab_brain_segmentation.ipynb`** — expects the raw `.mat` files from figshare:

```
MyDrive/
└── brain_segmentation_dataset/
    └── 1512427/
        ├── brainTumorDataPublic_1-766.zip
        ├── brainTumorDataPublic_767-1532.zip
        ├── brainTumorDataPublic_1533-2298.zip
        └── brainTumorDataPublic_2299-3064.zip
```

Dataset source: [Brain Tumor Dataset on figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

The notebook will unzip and convert the `.mat` files to PNG automatically on first run.

### 4. Change the data path variables

**`brain_tumor_segmentation.ipynb`** — update this variable to match your Drive path:

```python
DATA_ROOT = "/content/drive/MyDrive/brain_segmentation_dataset/archive"
```

**`matlab_brain_segmentation.ipynb`** — update these two variables:

```python
INPUT_DIR  = "/content/drive/MyDrive/brain_segmentation_dataset/1512427"   # raw .mat files
OUTPUT_DIR = "/content/drive/MyDrive/brain_segmentation_dataset/output_png" # converted PNGs saved here
```

### 5. Enable GPU

Go to **Runtime → Change runtime type → T4 GPU** before running. Both notebooks require a GPU.

### 6. Run all cells

**Runtime → Run all** (or `Ctrl+F9`). Runtime is roughly 30–60 minutes for 12 epochs on a T4 GPU.

---

## What the notebooks do

1. Mount Google Drive and verify GPU
2. Install dependencies (`segmentation-models-pytorch`, `albumentations`, `torchmetrics`, etc.)
3. EDA — class distribution, tumor size distribution, sample visualizations
4. Dataset class + augmentation pipeline (resize, flip, rotate, brightness)
5. Train/val split (85/15), DataLoader setup
6. U-Net (ResNet34 encoder, ImageNet weights) with combined BCE + Dice loss
7. Training loop — 12 epochs, AdamW optimizer, ReduceLROnPlateau scheduler
8. Visualize predictions — heatmaps, mask overlays, doctor border vs model contour comparison
9. Save model weights to Google Drive

---

## Dataset details

### Kaggle dataset (`archive/`)
- 3,064 PNG brain MRI images with corresponding binary tumor masks
- Images and masks share the same filename

### figshare 1512427 dataset
- 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients
- Three tumor types: **Meningioma** (708), **Glioma** (1,426), **Pituitary** (930)
- 512×512 resolution, stored as `.mat` files with `cjdata.image`, `cjdata.tumorMask`, and `cjdata.tumorBorder` fields
- Acquired at Nanfang Hospital and Tianjin Medical University, China (2005–2010)

> Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition." *PloS one* 10.10 (2015).

---

## Dependencies

Installed automatically inside the notebooks via `%pip install`:

```
segmentation-models-pytorch==0.3.1
timm==0.4.12
albumentations==1.3.0
torchmetrics==0.11.4
h5py==3.16.0   # matlab_brain_segmentation only
```