"""
Spectra Phase C — Look-alike Classifier
========================================
Lightweight CNN binary classifier that runs after U-Net segmentation.

Pipeline position:
    SAR patch → U-Net (pixel mask) → LookalikeClassifier (is this real oil?) → alert gate

Architecture: MobileNet-inspired lightweight CNN (~480K parameters).
"""

import argparse
import logging
import os
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = Path("models/lookalike_model.pth")
CONFIDENCE_THRESHOLD = 0.6  # below this → look-alike, suppress alert

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)))
        x = F.relu6(self.bn2(self.pw(x)))
        return x

class LookalikeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2), 
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LookalikeDataset(Dataset):
    def __init__(self, root: Path, augment: bool = True):
        self.samples: list[Tuple[Path, int]] = []
        oil_dir = root / "oil"
        non_dir = root / "non_oil"

        for d, label in [(oil_dir, 1), (non_dir, 0)]:
            if d.exists():
                for p in sorted(d.glob("*")):
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                        self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}. Ensure you ran --prep-kaggle successfully.")

        n_oil = sum(1 for _, l in self.samples if l == 1)
        n_non = len(self.samples) - n_oil
        logger.info("Dataset: %d oil, %d non-oil patches", n_oil, n_non)
        self.pos_weight = torch.tensor(n_non / max(n_oil, 1), dtype=torch.float32)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize((256, 256), Image.BILINEAR)
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def train(data_dir: Path, output_path: Path, epochs: int = 30, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    full_dataset = LookalikeDataset(data_dir, augment=True)
    val_size = max(1, int(len(full_dataset) * 0.15))
    train_ds, val_ds = random_split(full_dataset, [len(full_dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LookalikeNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=full_dataset.pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    if output_path.exists():
        logger.info(f"Loading existing weights from {output_path} for fine-tuning...")
        checkpoint = torch.load(output_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_acc = checkpoint.get("val_acc", 0.0)
    # -----------------------------------------------
    
    best_val_acc = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(1), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = (torch.sigmoid(model(imgs).squeeze(1)) >= CONFIDENCE_THRESHOLD).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = (correct / total) if total > 0 else 0
        logger.info(f"Epoch {epoch}/{epochs} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, output_path)
    
    return best_val_acc

# ---------------------------------------------------------------------------
# Inference & Prep Utilities
# ---------------------------------------------------------------------------

class LookalikeClassifier:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, threshold: float = CONFIDENCE_THRESHOLD):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = None
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = LookalikeNet().to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

    def classify(self, patch: np.ndarray) -> dict:
        if self.model is None: return {"lookalike_score": 1.0, "lookalike_passed": True}
        if patch.ndim == 3: patch = patch.mean(axis=2)
        img = Image.fromarray(patch.astype(np.float32)).resize((256, 256))
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = torch.sigmoid(self.model(tensor)).item()
        return {"lookalike_score": round(score, 4), "lookalike_passed": score >= self.threshold}

def prep_kaggle_dataset(kaggle_dir: Path, output_dir: Path):
    """Smart-match prep utility that ignores extensions and matches by stem."""
    img_search = list(kaggle_dir.rglob("images"))
    lbl_search = list(kaggle_dir.rglob("labels"))

    if not img_search or not lbl_search:
        raise FileNotFoundError(f"Missing 'images' or 'labels' in {kaggle_dir}")

    img_dir, lbl_dir = img_search[0], lbl_search[0]
    for d in ["oil", "non_oil"]: (output_dir / d).mkdir(parents=True, exist_ok=True)
    
    # Map all available masks by their "stem" (filename without extension)
    mask_map = {f.stem: f for f in lbl_dir.iterdir() if f.is_file()}
    
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    n_oil = n_non = 0
    logger.info(f"Scanning {len(image_files)} images...")

    for img_path in image_files:
        # Try to find a mask that matches the image stem
        # e.g., if image is 'patch1.jpg', looks for 'patch1' in the masks folder
        mask_path = mask_map.get(img_path.stem)
        
        if not mask_path:
            continue

        try:
            mask = np.array(Image.open(mask_path).convert("L"))
            # If mask has any white pixels (or > 1% for robustness)
            is_oil = (mask > 128).sum() / mask.size > 0.01
            
            dest = (output_dir / "oil") if is_oil else (output_dir / "non_oil")
            shutil.copy2(img_path, dest / img_path.name)
            
            if is_oil: n_oil += 1
            else: n_non += 1
        except Exception as e:
            logger.warning(f"Could not process {img_path.name}: {e}")
    
    logger.info(f"Prep complete: {n_oil} oil, {n_non} non-oil -> {output_dir}")
    return n_oil, n_non
# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Spectra Look-alike Classifier")
    
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--prep-kaggle", action="store_true")
    parser.add_argument("--test", action="store_true")
    
    parser.add_argument("--kaggle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/lookalike_dataset"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/lookalike_dataset"))
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)  # <--- ADD THIS LINE
    parser.add_argument("--image", type=Path)

    args = parser.parse_args()

    if args.prep_kaggle:
        if not args.kaggle_dir:
            print("Error: --kaggle-dir is required for prep.")
        else:
            prep_kaggle_dataset(args.kaggle_dir, args.output_dir)
    elif args.train:
        # Pass the lr argument to the train function
        train(args.data_dir, args.output, args.epochs, lr=args.lr) # <--- UPDATE THIS LINE
    # ... rest of the code

    if args.prep_kaggle:
        if not args.kaggle_dir:
            print("Error: --kaggle-dir is required for prep.")
        else:
            prep_kaggle_dataset(args.kaggle_dir, args.output_dir)
    elif args.train:
        train(args.data_dir, args.output, args.epochs)
    elif args.test and args.image:
        res = LookalikeClassifier(args.output).classify(np.array(Image.open(args.image)))
        print(f"Result: {res}")
    else:
        parser.print_help()