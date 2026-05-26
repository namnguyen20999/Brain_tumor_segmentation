# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
import sys
print(sys.executable)
!nvidia-smi


# %%
%pip install -q --upgrade pip
%pip install -q segmentation-models-pytorch==0.3.1
%pip install -q timm==0.4.12
%pip install -q albumentations==1.3.0
%pip install -q torchmetrics==0.11.4
%pip install -q h5py==3.16.0


# %%
import os
import zipfile
import numpy as np
import scipy.io
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from PIL import Image

# %% [markdown]
#  ## Step 0 — Inspect a single .mat file
# 
#  Before converting anything, open one file and understand
# 
#  what fields it contains and what they look like.

# %%
SAMPLE_MAT = "/content/drive/MyDrive/brain_segmentation_dataset/1512427/brainTumorDataPublic_1-766/1.mat"

with h5py.File(SAMPLE_MAT, "r") as f:
    cjdata = f["cjdata"]

    print("Fields inside cjdata:")
    for field in cjdata.keys():
        val = np.array(cjdata[field])
        print(f"  {field:15s} → shape: {val.shape}, dtype: {val.dtype}")

# %%
with h5py.File(SAMPLE_MAT, "r") as f:
    cjdata = f["cjdata"]

    label        = int(np.array(cjdata["label"]).flat[0])
    image        = np.array(cjdata["image"],     dtype=np.float64).T
    tumor_mask   = np.array(cjdata["tumorMask"], dtype=np.uint8).T
    tumor_border = np.array(cjdata["tumorBorder"]).flatten()

    # PID is stored as unicode char codes in h5py
    pid_raw = np.array(cjdata["PID"]).flatten()
    pid     = "".join(chr(int(c)) for c in pid_raw)

CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
print(f"Patient ID  : {pid}")
print(f"Label       : {label} → {CLASS_NAMES[label]}")
print(f"Image shape : {image.shape}  dtype: {image.dtype}")
print(f"Image range : [{image.min():.1f}, {image.max():.1f}]")
print(f"Mask shape  : {tumor_mask.shape}  unique values: {np.unique(tumor_mask)}")
print(f"Border pts  : {len(tumor_border)//2} (x,y) points")

# %%
# Normalize image to [0,255] for display
lo, hi   = image.min(), image.max()
im_uint8 = (255.0 / (hi - lo) * (image - lo)).astype(np.uint8)

# tumorBorder: [x1,y1,x2,y2,...] → reshape to [[x1,y1],[x2,y2],...]
# x = column (horizontal), y = row (vertical)
border_coords = tumor_border.reshape(-1, 2)
bx, by        = border_coords[:, 0], border_coords[:, 1]

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"Single sample — {CLASS_NAMES[label]} (PID: {pid})", fontsize=14)

axs[0].imshow(im_uint8, cmap="gray")
axs[0].set_title("MRI image\n(normalized to 0-255)")
axs[0].axis("off")

axs[1].imshow(tumor_mask, cmap="gray")
axs[1].set_title("tumorMask\n(1=tumor, 0=background)")
axs[1].axis("off")

axs[2].imshow(im_uint8, cmap="gray")
axs[2].plot(bx, by, "r-", linewidth=1.5, label="border")
axs[2].scatter(bx[0], by[0], c="lime", s=50, zorder=5, label="start point")
axs[2].set_title("tumorBorder\n(doctor's hand-drawn contour)")
axs[2].legend(fontsize=8)
axs[2].axis("off")

axs[3].imshow(im_uint8, cmap="gray")
axs[3].imshow(tumor_mask, cmap="Reds", alpha=0.4)
axs[3].plot(bx, by, "cyan", linewidth=1.5, label="border")
axs[3].set_title("Overlay\n(image + mask + border)")
axs[3].legend(fontsize=8)
axs[3].axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
#  ## Step 1 — EDA: scan all .mat files
# 
#  Collect label, tumor size, image stats from every sample
# 
#  to understand the full dataset before training.

# %%
def load_mat(mat_path):
    with h5py.File(mat_path, "r") as f:
        cjdata = f["cjdata"]
        label  = int(np.array(cjdata["label"]).flat[0])
        image  = np.array(cjdata["image"],       dtype=np.float64).T
        mask   = np.array(cjdata["tumorMask"],   dtype=np.uint8).T
        border = np.array(cjdata["tumorBorder"]).flatten()
    return label, image, mask, border

INPUT_DIR = "/content/drive/MyDrive/brain_segmentation_dataset/1512427"

records = []

for root, dirs, files in os.walk(INPUT_DIR):
    if not os.path.basename(root).startswith("brainTumorDataPublic_"):
        continue
    for fname in sorted(files):
        if not fname.endswith(".mat"):
            continue
        try:
            label, image, mask, _ = load_mat(os.path.join(root, fname))
            tumor_ratio = mask.sum() / mask.size
            records.append({
                "label":       label,
                "tumor_ratio": tumor_ratio,
                "img_mean":    image.mean(),
                "img_std":     image.std(),
            })
        except Exception as e:
            print(f"Skipped {fname}: {e}")

print(f"Total samples loaded: {len(records)}")

# %%
import seaborn as sns

labels      = [r["label"]       for r in records]
ratios      = [r["tumor_ratio"] for r in records]
img_means   = [r["img_mean"]    for r in records]
img_stds    = [r["img_std"]     for r in records]

# --- Class distribution ---
class_counts = {CLASS_NAMES[k]: labels.count(k) for k in CLASS_NAMES}
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Dataset Overview", fontsize=14)

axs[0].bar(class_counts.keys(), class_counts.values(),
           color=["steelblue", "tomato", "mediumseagreen"])
for i, (k, v) in enumerate(class_counts.items()):
    axs[0].text(i, v + 10, str(v), ha="center", fontsize=11)
axs[0].set_title("Class distribution")
axs[0].set_ylabel("Number of images")

# --- Tumor size distribution per class ---
for k, name in CLASS_NAMES.items():
    class_ratios = [r["tumor_ratio"] for r in records if r["label"] == k]
    axs[1].hist(class_ratios, bins=30, alpha=0.6, label=name)
axs[1].set_title("Tumor area ratio per class")
axs[1].set_xlabel("Tumor / total pixels")
axs[1].set_ylabel("Count")
axs[1].legend()

# --- Image intensity distribution ---
axs[2].hist(img_means, bins=40, color="slategray", alpha=0.8)
axs[2].set_title("Image mean intensity distribution")
axs[2].set_xlabel("Mean pixel value")
axs[2].set_ylabel("Count")

plt.tight_layout()
plt.show()

print("\nTumor area ratio stats:")
for k, name in CLASS_NAMES.items():
    r = [x["tumor_ratio"] for x in records if x["label"] == k]
    print(f"  {name:12s} → mean: {np.mean(r)*100:.2f}%  min: {np.min(r)*100:.2f}%  max: {np.max(r)*100:.2f}%")


# %%
# --- Show one sample per class with tumorBorder overlay ---
fig, axs = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("One sample per class — image / mask / border / overlay", fontsize=14)

for row, (k, name) in enumerate(CLASS_NAMES.items()):
    # Find first file with this label
    for root, dirs, files in os.walk(INPUT_DIR):
        if not os.path.basename(root).startswith("brainTumorDataPublic_"):
            continue
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            label, image, mask, border = load_mat(os.path.join(root, fname))
            if label != k:
                continue

            lo, hi   = image.min(), image.max()
            im_u8    = (255.0 / (hi - lo) * (image - lo)).astype(np.uint8)
            bcoords  = border.reshape(-1, 2)
            bx, by   = bcoords[:, 0], bcoords[:, 1]

            axs[row, 0].imshow(im_u8, cmap="gray")
            axs[row, 0].set_title(f"{name}\nMRI image")
            axs[row, 0].axis("off")

            axs[row, 1].imshow(mask, cmap="gray")
            axs[row, 1].set_title("tumorMask")
            axs[row, 1].axis("off")

            axs[row, 2].imshow(im_u8, cmap="gray")
            axs[row, 2].plot(bx, by, "r-", linewidth=1.5)
            axs[row, 2].scatter(bx[0], by[0], c="lime", s=50, zorder=5)
            axs[row, 2].set_title("tumorBorder (doctor)")
            axs[row, 2].axis("off")

            axs[row, 3].imshow(im_u8, cmap="gray")
            axs[row, 3].imshow(mask, cmap="Reds", alpha=0.4)
            axs[row, 3].plot(bx, by, "cyan", linewidth=1.5)
            axs[row, 3].set_title("Overlay")
            axs[row, 3].axis("off")
            break
        else:
            continue
        break

plt.tight_layout()
plt.show()


# %% [markdown]
#  ## Step 2 — Convert .mat → PNG (images + masks)
# 
#  Now that we understand the data, convert everything to PNG.
# 
#  We save images/ and masks/ as flat folders so they mirror
# 
#  the archive/ dataset structure used in the original code.

# %%
OUTPUT_DIR = "/content/drive/MyDrive/brain_segmentation_dataset/output_png"
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"),  exist_ok=True)

# Unzip if not already done
for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".zip"):
        zip_path    = os.path.join(INPUT_DIR, fname)
        extract_dir = os.path.join(INPUT_DIR, fname.replace(".zip", ""))
        if not os.path.exists(extract_dir):
            print(f"Unzipping {fname}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        else:
            print(f"Already extracted: {fname}")

converted, skipped = 0, 0
for root, dirs, files in os.walk(INPUT_DIR):
    if not os.path.basename(root).startswith("brainTumorDataPublic_"):
        continue
    for fname in sorted(files):
        if not fname.endswith(".mat"):
            continue
        mat_path = os.path.join(root, fname)
        try:
            label, im, mask, _ = load_mat(mat_path)

            lo, hi   = im.min(), im.max()
            im_uint8 = (255.0 / (hi - lo) * (im - lo)).astype(np.uint8)
            mask_uint8 = (mask > 0).astype(np.uint8) * 255

            base   = os.path.splitext(fname)[0]
            parent = os.path.basename(root)

            Image.fromarray(im_uint8).save(os.path.join(OUTPUT_DIR, "images", f"{parent}_{base}.png"))
            Image.fromarray(mask_uint8).save(os.path.join(OUTPUT_DIR, "masks",  f"{parent}_{base}.png"))
            converted += 1
        except Exception as e:
            print(f"Error: {mat_path}: {e}")
            skipped += 1

print(f"\nDone. Converted: {converted}, Skipped: {skipped}")


# %% [markdown]
#  ## Step 3 — Dataset class & transforms
# 
#  BrainTumorSegDataset loads image+mask pairs from disk.
# 
#  train_transform applies augmentation; valid_transform does not.

# %%
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torchmetrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

images_dir = Path(OUTPUT_DIR) / "images"
masks_dir  = Path(OUTPUT_DIR) / "masks"


class BrainTumorSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transforms  = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 127).astype("float32")
        if self.transforms:
            aug  = self.transforms(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"].unsqueeze(0)
        else:
            img  = TF.to_tensor(img)
            mask = torch.tensor(mask).unsqueeze(0)
        return img, mask


train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Normalize(),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])


# %%
# Visualize augmentation effect: show same image before and after transforms
sample_img  = np.array(Image.open(sorted(images_dir.glob("*.png"))[0]).convert("RGB"))
sample_mask = np.array(Image.open(sorted(masks_dir.glob("*.png"))[0]).convert("L"))
sample_mask_bin = (sample_mask > 127).astype(np.float32)

fig, axs = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle("Augmentation preview — same image put through train_transform 5 times", fontsize=13)

# Original (no transform)
axs[0, 0].imshow(sample_img, cmap="gray")
axs[0, 0].set_title("Original image")
axs[0, 0].axis("off")
axs[1, 0].imshow(sample_mask_bin, cmap="gray")
axs[1, 0].set_title("Original mask")
axs[1, 0].axis("off")

# 4 random augmentations
for col in range(1, 5):
    aug  = train_transform(image=sample_img, mask=sample_mask_bin)
    # Denormalize image for display
    img_show = aug["image"].permute(1, 2, 0).numpy()
    img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_show = np.clip(img_show, 0, 1)
    axs[0, col].imshow(img_show)
    axs[0, col].set_title(f"Augmented #{col}")
    axs[0, col].axis("off")
    axs[1, col].imshow(aug["mask"].numpy(), cmap="gray")
    axs[1, col].set_title(f"Mask #{col}")
    axs[1, col].axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
#  ## Step 4 — Train/val split & dataloaders

# %%
from sklearn.model_selection import train_test_split

all_images = sorted(list(images_dir.glob("*.png")))
all_masks  = [masks_dir / img.name for img in all_images]

missing = [m for m in all_masks if not m.exists()]
print(f"Missing masks : {len(missing)}")
print(f"Total images  : {len(all_images)}")

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    all_images, all_masks, test_size=0.15, random_state=SEED
)
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)}")

BATCH_SIZE   = 8
train_ds     = BrainTumorSegDataset(train_imgs, train_masks, transforms=train_transform)
val_ds       = BrainTumorSegDataset(val_imgs,   val_masks,   transforms=valid_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# %%
# Visualize one batch straight from the dataloader
# This confirms shapes and values are correct before training
imgs_batch, masks_batch = next(iter(train_loader))
print(f"Image batch : {imgs_batch.shape}  range [{imgs_batch.min():.2f}, {imgs_batch.max():.2f}]")
print(f"Mask batch  : {masks_batch.shape}  unique values: {masks_batch.unique().tolist()}")

fig, axs = plt.subplots(2, BATCH_SIZE, figsize=(22, 6))
fig.suptitle("One training batch (top=image, bottom=mask)", fontsize=13)
for i in range(BATCH_SIZE):
    img_show = imgs_batch[i].permute(1, 2, 0).numpy()
    img_show = img_show * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_show = np.clip(img_show, 0, 1)
    axs[0, i].imshow(img_show)
    axs[0, i].axis("off")
    axs[1, i].imshow(masks_batch[i].squeeze(), cmap="gray")
    axs[1, i].axis("off")
plt.tight_layout()
plt.show()


# %% [markdown]
#  ## Step 5 — Model, loss, optimizer, metrics

# %%
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}")


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds        = torch.sigmoid(preds)
        preds        = preds.view(preds.size(0), -1)
        targets      = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(1)
        union        = preds.sum(1) + targets.sum(1)
        dice         = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


bce  = nn.BCEWithLogitsLoss()
dice = DiceLoss()

def loss_fn(preds, targets):
    return 0.5 * bce(preds, targets) + 0.5 * dice(preds, targets)

optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
metric_iou = torchmetrics.JaccardIndex(task="binary").to(DEVICE)


# %%
# Sanity check: run one batch through the untrained model
# to confirm shapes are correct before starting training
model.eval()
with torch.no_grad():
    test_imgs  = imgs_batch[:2].to(DEVICE)
    test_preds = model(test_imgs)
    test_probs = torch.sigmoid(test_preds)
print(f"Input shape : {test_imgs.shape}")
print(f"Output shape: {test_preds.shape}")
print(f"Output range (logits): [{test_preds.min():.2f}, {test_preds.max():.2f}]")
print(f"Output range (probs) : [{test_probs.min():.2f}, {test_probs.max():.2f}]")


# %% [markdown]
#  ## Step 6 — Training & validation loops

# %%
from tqdm.auto import tqdm


def train_one_epoch(loader, model, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds       = model(imgs)
        loss        = loss_fn(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def valid_one_epoch(loader, model):
    model.eval()
    running_loss = 0.0
    iou_score    = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds       = model(imgs)
            loss        = loss_fn(preds, masks)
            running_loss += loss.item() * imgs.size(0)
            preds_bin = (torch.sigmoid(preds) > 0.5).int()
            for p, t in zip(preds_bin, masks.int()):
                iou_score += metric_iou(p.squeeze(0), t.squeeze(0)).item()
    return running_loss / len(loader.dataset), iou_score / len(loader.dataset)


EPOCHS  = 12
history = {"train_loss": [], "val_loss": [], "val_iou": [], "lr": []}

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}  (lr={optimizer.param_groups[0]['lr']:.6f})")
    tr_loss           = train_one_epoch(train_loader, model, optimizer)
    val_loss, val_iou = valid_one_epoch(val_loader,   model)
    scheduler.step(val_loss)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)
    history["lr"].append(optimizer.param_groups[0]["lr"])
    print(f"  Train loss: {tr_loss:.4f}  Val loss: {val_loss:.4f}  Val IoU: {val_iou:.4f}")


# %%
# Training curves
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle("Training history", fontsize=13)

axs[0].plot(history["train_loss"], label="train", marker="o")
axs[0].plot(history["val_loss"],   label="val",   marker="o")
axs[0].set_title("Loss (BCE + Dice)")
axs[0].set_xlabel("Epoch")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(history["val_iou"], color="green", marker="o")
axs[1].set_title("Validation IoU")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("IoU")
axs[1].grid(True)

axs[2].plot(history["lr"], color="orange", marker="o")
axs[2].set_title("Learning rate")
axs[2].set_xlabel("Epoch")
axs[2].set_yscale("log")
axs[2].grid(True)

plt.tight_layout()
plt.show()

print(f"\nBest Val IoU  : {max(history['val_iou']):.4f} at epoch {history['val_iou'].index(max(history['val_iou']))+1}")
print(f"Final Val loss: {history['val_loss'][-1]:.4f}")


# %% [markdown]
#  ## Step 7 — Visualize predictions with doctor border overlay

# %%
import cv2


def overlay_mask(image, mask, alpha=0.4, color=(255, 0, 0)):
    img        = image.copy()
    mask_layer = np.zeros_like(img)
    for c, v in enumerate(color):
        mask_layer[:, :, c] = (mask * v).astype(np.uint8)
    return cv2.addWeighted(img, 1 - alpha, mask_layer, alpha, 0)


def visualize_predictions(model, img_paths, mask_paths, n_samples=6):
    model.eval()
    # 5 columns: image | ground truth | prediction heatmap | prediction overlay | contours
    fig, axs = plt.subplots(n_samples, 5, figsize=(24, 5 * n_samples))
    fig.suptitle("Prediction results", fontsize=14)

    col_titles = ["MRI image", "Ground truth mask", "Pred heatmap", "Pred overlay", "Pred contours"]
    for col, title in enumerate(col_titles):
        axs[0, col].set_title(title, fontsize=11)

    for i in range(n_samples):
        img_np  = np.array(Image.open(img_paths[i]).convert("RGB").resize((256, 256)))
        mask_np = np.array(Image.open(mask_paths[i]).convert("L").resize((256, 256))) > 127

        input_t = valid_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_prob = torch.sigmoid(model(input_t)).squeeze().cpu().numpy()
        pred_bin    = (pred_prob > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = img_np.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        iou_val = np.logical_and(pred_bin, mask_np).sum() / (np.logical_or(pred_bin, mask_np).sum() + 1e-6)

        axs[i, 0].imshow(img_np, cmap="gray")
        axs[i, 0].set_ylabel(f"IoU={iou_val:.2f}", fontsize=9)

        axs[i, 1].imshow(mask_np, cmap="gray")

        axs[i, 2].imshow(pred_prob, cmap="hot")
        plt.colorbar(axs[i, 2].images[0], ax=axs[i, 2], fraction=0.046)

        axs[i, 3].imshow(overlay_mask(img_np, pred_bin))

        axs[i, 4].imshow(contour_img)

        for col in range(5):
            axs[i, col].axis("off")

    plt.tight_layout()
    plt.show()


visualize_predictions(model, val_imgs, val_masks, n_samples=6)


# %%
# Compare ground truth border vs predicted border side by side
# Load the original .mat files for the val samples to get tumorBorder
def get_mat_path_from_png_name(png_name, input_dir):
    # png name format: brainTumorDataPublic_1-766_1.png
    # → folder: brainTumorDataPublic_1-766, file: 1.mat
    parts  = png_name.replace(".png", "")
    # split on last underscore to get folder and base
    idx    = parts.rfind("_")
    folder = parts[:idx]
    base   = parts[idx+1:] + ".mat"
    return os.path.join(input_dir, folder, base)

n_border = 6
fig, axs = plt.subplots(n_border, 3, figsize=(15, 5 * n_border))
fig.suptitle("Ground truth border (doctor) vs predicted contour", fontsize=14)

for i in range(n_border):
    img_np  = np.array(Image.open(val_imgs[i]).convert("RGB").resize((256, 256)))
    mask_np = np.array(Image.open(val_masks[i]).convert("L").resize((256, 256))) > 127

    input_t = valid_transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_prob = torch.sigmoid(model(input_t)).squeeze().cpu().numpy()
    pred_bin = (pred_prob > 0.5).astype(np.uint8)

    # Load tumorBorder from original .mat
    mat_path = get_mat_path_from_png_name(val_imgs[i].name, INPUT_DIR)
    try:
        _, orig_img, _, border = load_mat(mat_path)
        lo, hi    = orig_img.min(), orig_img.max()
        orig_u8   = (255.0 / (hi - lo) * (orig_img - lo)).astype(np.uint8)
        bcoords   = border.reshape(-1, 2)
        bx, by    = bcoords[:, 0], bcoords[:, 1]
        # Scale border coords from original image size to 256x256
        scale_x   = 256 / orig_img.shape[1]
        scale_y   = 256 / orig_img.shape[0]
        bx_scaled = bx * scale_x
        by_scaled = by * scale_y
        has_border = True
    except Exception:
        has_border = False

    # Doctor's border
    axs[i, 0].imshow(img_np, cmap="gray")
    if has_border:
        axs[i, 0].plot(bx_scaled, by_scaled, "r-", linewidth=1.5, label="doctor border")
        axs[i, 0].legend(fontsize=8)
    axs[i, 0].set_title("Doctor's tumorBorder")
    axs[i, 0].axis("off")

    # Model prediction contour
    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    axs[i, 1].imshow(img_np, cmap="gray")
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            axs[i, 1].plot(cnt[:, 0], cnt[:, 1], "lime", linewidth=1.5, label="predicted")
    axs[i, 1].legend(fontsize=8)
    axs[i, 1].set_title("Model predicted contour")
    axs[i, 1].axis("off")

    # Both overlaid
    axs[i, 2].imshow(img_np, cmap="gray")
    if has_border:
        axs[i, 2].plot(bx_scaled, by_scaled, "r-", linewidth=1.5, label="doctor")
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim == 2:
            axs[i, 2].plot(cnt[:, 0], cnt[:, 1], "lime", linewidth=1.5, label="model")
    axs[i, 2].legend(fontsize=8)
    axs[i, 2].set_title("Doctor vs model")
    axs[i, 2].axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
#  ## Step 8 — Save model

# %%
torch.save(
    model.state_dict(),
    "/content/drive/MyDrive/brain_segmentation_dataset/brain_tumor_unet_1512427.pth"
)
print("Model saved!")

# %%



