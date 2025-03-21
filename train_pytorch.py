import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from models import EVA02Transformer, eva02_large

class TFRecordDataset(Dataset):
    def __init__(self, tfrecord_path, transform=None, meta_path=None):
        self.tfrecord_path = tfrecord_path
        self.transform = transform
        if not os.path.exists(tfrecord_path):
            raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")
        if not meta_path or not os.path.exists(meta_path):
            raise ValueError("meta_path is required and must exist")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.num_classes = meta["num_classes"]
        self.dataset = tf.data.TFRecordDataset(tfrecord_path)
        self.feature_description = {
            "image_bytes": tf.io.FixedLenFeature([], tf.string),
            "label_indexes": tf.io.VarLenFeature(tf.int64),
        }
        self.data = [self._parse_tfrecord(item) for item in self.dataset]

    def _parse_tfrecord(self, example_proto):
        example = tf.io.parse_single_example(example_proto, self.feature_description)
        image = tf.io.decode_jpeg(example["image_bytes"], channels=3)
        labels = tf.sparse.to_dense(example["label_indexes"]).numpy()
        return image.numpy(), labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, labels = self.data[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        if image.shape[0] != 3:
            raise ValueError(f"Image channels must be 3, got {image.shape}")
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        if not all(0 <= idx < self.num_classes for idx in labels):
            raise ValueError(f"Label indexes {labels} out of range [0, {self.num_classes-1}]")
        label_tensor[labels] = 1
        return image, label_tensor

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    num_workers = 4 if os.name != 'nt' else 0  # Windows 不支持多进程
    print("Loading training dataset...")
    train_dataset = TFRecordDataset(
        os.path.join(args.dataset_root, "record_shards_train/aibooru_train.tfrecord"),
        transform,
        meta_path=os.path.join(args.dataset_root, "aibooru.json")
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    print("Loading validation dataset...")
    val_dataset = TFRecordDataset(
        os.path.join(args.dataset_root, "record_shards_val/aibooru_val.tfrecord"),
        transform,
        meta_path=os.path.join(args.dataset_root, "aibooru.json")
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    print("Building model...")
    model_builder = eva02_large()
    model = model_builder.build(num_classes=num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            if torch.isnan(images).any() or torch.isinf(images).any():
                raise ValueError("Input images contain NaN or Inf values")
            if images.shape[1] != 3 or images.shape[2:] != (args.image_size, args.image_size):
                raise ValueError(f"Expected image shape [B, 3, {args.image_size}, {args.image_size}], got {images.shape}")
            if labels.shape[1] != num_classes:
                raise ValueError(f"Expected label shape [B, {num_classes}], got {labels.shape}")
            optimizer.zero_grad()
            outputs = model(images)
            if outputs.shape[1] != num_classes:
                raise ValueError(f"Model output shape [B, {num_classes}] expected, got {outputs.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "eva02_large_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

        torch.save(model.state_dict(), os.path.join(args.output_dir, f"eva02_large_epoch_{epoch+1}.pth"))

    config = {
        "model_name": "eva02_large",
        "image_size": args.image_size,
        "model_args": {
            "num_classes": num_classes,
            "patch_size": 16,
            "num_layers": 24,
            "embed_dim": 1024,
            "mlp_dim": (1024 * 4 * 2) // 3,
            "num_heads": 16,
            "scale_mlp": True
        }
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training completed! Best model saved at {os.path.join(args.output_dir, 'eva02_large_best.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EVA-02 Large model")
    parser.add_argument("--dataset_root", type=str, default="/home/smilingwolf/datasets", help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    train_model(args)