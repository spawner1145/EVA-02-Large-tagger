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
        # Limit parsing for faster initialization during testing/dev
        # self.data = [self._parse_tfrecord(item) for item in self.dataset.take(100)] # Uncomment for faster testing
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
        # Ensure image has 3 channels after transform
        if image.shape[0] != 3:
             # Handle grayscale images that might become single channel
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1) # Repeat channel 3 times
            else:
                raise ValueError(f"Image channels must be 3, got {image.shape}")

        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        # Ensure labels are within valid range
        valid_labels = [l for l in labels if 0 <= l < self.num_classes]
        if len(valid_labels) != len(labels):
            print(f"Warning: Invalid label indices found: {labels}. Using only valid ones: {valid_labels}")
            # Optionally raise an error or handle differently
            # raise ValueError(f"Label indexes {labels} out of range [0, {self.num_classes-1}]")
        if valid_labels:
             label_tensor[valid_labels] = 1
        return image, label_tensor

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    num_workers = 4 if os.name != 'nt' else 0  # Windows compatibility
    print("Loading training dataset...")
    train_dataset = TFRecordDataset(
        os.path.join(args.dataset_root, "record_shards_train/aibooru_train.tfrecord"),
        transform,
        meta_path=os.path.join(args.dataset_root, "aibooru.json")
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device == 'cuda' else False)

    print("Loading validation dataset...")
    val_dataset = TFRecordDataset(
        os.path.join(args.dataset_root, "record_shards_val/aibooru_val.tfrecord"),
        transform,
        meta_path=os.path.join(args.dataset_root, "aibooru.json")
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if device == 'cuda' else False)

    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    print("Building model...")
    model_builder = eva02_large()
    model = model_builder.build(num_classes=num_classes)

    # --- Modification for Retraining Start ---
    start_epoch = 0
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            print(f"Loading checkpoint '{args.resume_checkpoint}'")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            # Adjust for DataParallel wrapper if necessary
            state_dict = checkpoint
            # If the checkpoint was saved with DataParallel, keys might start with 'module.'
            if isinstance(model, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif not isinstance(model, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False) # Use strict=False if architecture changed slightly
            print(f"Loaded checkpoint '{args.resume_checkpoint}'")
            # Optionally load optimizer state and epoch number if saved in checkpoint
            # if 'epoch' in checkpoint:
            #     start_epoch = checkpoint['epoch']
            # if 'optimizer' in checkpoint:
            #     optimizer.load_state_dict(checkpoint['optimizer'])
            # if 'scheduler' in checkpoint:
            #     scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print(f"Warning: Checkpoint file not found at '{args.resume_checkpoint}'. Starting from scratch.")
    # --- Modification for Retraining End ---

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    print(f"Starting training from epoch {start_epoch + 1}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad() # Reset gradients at the beginning of epoch
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training")):
            images, labels = images.to(device), labels.to(device)

            # Input validation (optional but recommended)
            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"Warning: NaN or Inf found in input images at batch {i}. Skipping batch.")
                continue
            if images.shape[1] != 3 or images.shape[2:] != (args.image_size, args.image_size):
                 print(f"Warning: Unexpected image shape {images.shape} at batch {i}. Expected [B, 3, {args.image_size}, {args.image_size}]. Skipping batch.")
                 continue
            if labels.shape[1] != num_classes:
                 print(f"Warning: Unexpected label shape {labels.shape} at batch {i}. Expected [B, {num_classes}]. Skipping batch.")
                 continue

            outputs = model(images)
            # Output validation
            if outputs.shape[1] != num_classes:
                raise ValueError(f"Model output shape [B, {num_classes}] expected, got {outputs.shape}")
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN or Inf found in model outputs at batch {i}. Skipping loss calculation for this batch.")
                continue # Skip backpropagation if output is invalid

            loss = criterion(outputs, labels)

            # Gradient accumulation (optional, if batch size is effectively larger)
            # loss = loss / accumulation_steps
            loss.backward()

            # Gradient clipping (optional)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step optimizer periodically if using gradient accumulation
            # if (i + 1) % accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            optimizer.step() # Standard optimizer step
            optimizer.zero_grad() # Reset gradients for next iteration

            train_loss += loss.item()

        # Final optimizer step if using accumulation and loop ends before a step
        # if (len(train_loader) % accumulation_steps != 0):
        #     optimizer.step()
        #     optimizer.zero_grad()

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
        print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)

        # Save checkpoint logic
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'num_classes': num_classes # Save num_classes for consistency check
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(args.output_dir, "eva02_large_best.pth")
            torch.save(checkpoint_data, best_model_path)
            print(f"Saved best model checkpoint to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

        # Save epoch checkpoint
        epoch_model_path = os.path.join(args.output_dir, f"eva02_large_epoch_{epoch+1}.pth")
        torch.save(checkpoint_data, epoch_model_path)
        print(f"Saved epoch checkpoint to {epoch_model_path}")

    # Save final config
    config = {
        "model_name": "eva02_large",
        "image_size": args.image_size,
        "num_classes": num_classes,
        "model_args": {
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

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}. Best model saved at {os.path.join(args.output_dir, 'eva02_large_best.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain EVA-02 Large model from checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root containing TFRecord shards and aibooru.json")
    parser.add_argument("--output_dir", type=str, default="./retrain_checkpoints", help="Directory to save new checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to the checkpoint file to resume training from (.pth)")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for training (must match original training if resuming)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of additional epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (consider using a smaller LR for retraining)")

    args = parser.parse_args()

    # Basic validation
    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(f"Dataset root directory not found: {args.dataset_root}")
    if args.resume_checkpoint and not os.path.isfile(args.resume_checkpoint):
         print(f"Warning: resume_checkpoint path specified but file not found: {args.resume_checkpoint}. Training will start from scratch.")
         args.resume_checkpoint = None # Treat as starting from scratch

    train_model(args)