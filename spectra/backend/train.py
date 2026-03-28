import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw/oil-spill")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


class OilSpillDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image — convert to grayscale, duplicate to 2 channels (simulates VV+VH)
        img = Image.open(self.image_paths[idx]).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.stack([img, img], axis=0)  # shape: (2, 256, 256)

        # Load mask — binary: spill=1, no spill=0
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((IMG_SIZE, IMG_SIZE))
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)  # binarize
        mask = np.expand_dims(mask, axis=0)   # shape: (1, 256, 256)

        return torch.tensor(img), torch.tensor(mask)


def get_dataloaders():
    image_dir = DATA_DIR / "train" / "images"
    mask_dir = DATA_DIR / "train" / "labels"

    image_paths = sorted(image_dir.glob("*.jpg"))
    mask_paths = sorted(mask_dir.glob("*.png"))

    assert len(image_paths) == len(mask_paths), "Image/mask count mismatch"
    print(f"Total samples: {len(image_paths)}")

    # 80/20 train/val split
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_ds = OilSpillDataset(train_imgs, train_masks)
    val_ds = OilSpillDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def train():
    train_loader, val_loader = get_dataloaders()

    # U-Net with ResNet34 encoder, 2 input channels, 1 output class
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=2,
        classes=1,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")

    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = bce_loss(preds, masks) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = bce_loss(preds, masks) + dice_loss(preds, masks)
                val_loss += loss.item()
                val_iou += iou_score(preds, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val IoU: {avg_val_iou:.4f}")

        # Save best model
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), MODELS_DIR / "spectra_model.pth")
            print(f"  --> Best model saved (IoU: {best_iou:.4f})")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Model saved to models/spectra_model.pth")


if __name__ == "__main__":
    train()