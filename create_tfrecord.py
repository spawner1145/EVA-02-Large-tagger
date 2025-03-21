import argparse
import json
import os
from pathlib import Path
import logging
from io import BytesIO

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TFRecord 特征定义函数
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# 创建单个 TFRecord 示例
def create_example(image_path, label_indexes, img_size=None):
    try:
        with Image.open(image_path) as img:
            img.verify()
            img = img.convert("RGB")
            if img_size:
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()
            if not image_bytes:
                raise ValueError("Image bytes are empty")
    except Exception as e:
        logging.warning(f"Skipping corrupted image {image_path}: {str(e)}")
        return None
    if not label_indexes:  # 检查空标签
        logging.warning(f"Empty label indexes for {image_path}")
        return None
    if not all(isinstance(idx, int) and idx >= 0 for idx in label_indexes):
        raise ValueError(f"Label indexes must be non-negative integers, got {label_indexes}")
    feature = {
        "image_bytes": _bytes_feature(image_bytes),
        "label_indexes": _int64_feature(label_indexes),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

# 更新 selected_tags.csv 文件
def update_tags_csv(tags_csv, new_labels):
    """更新 selected_tags.csv，添加新标签并设为类别 0"""
    tags_df = pd.read_csv(tags_csv)
    if 'name' not in tags_df.columns:
        raise ValueError(f"{tags_csv} must contain 'name' column")
    
    # 将 name 列转为集合，用于检查标签是否已存在
    existing_labels = set(tags_df["name"].astype(str).tolist())
    
    # 只添加不存在的标签
    tags_to_add = [label for label in new_labels if label and label not in existing_labels]
    
    if tags_to_add:
        logging.info(f"New labels to add: {tags_to_add}")
        new_rows = pd.DataFrame({"name": tags_to_add, "category": [0] * len(tags_to_add)})
        tags_df = pd.concat([tags_df, new_rows], ignore_index=True)
        tags_df.to_csv(tags_csv, index=False)
        logging.info(f"Added {len(tags_to_add)} new labels to {tags_csv}")
    else:
        logging.info(f"No new labels to add; all tags already exist in {tags_csv}")
    return tags_df

# 主函数：创建 TFRecord 文件
def create_tfrecords(dataset_folder, output_path, split_ratio=0.7, img_size=224, tags_csv="selected_tags.csv"):
    dataset_folder = Path(dataset_folder)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 验证并读取 tags_csv
    if not Path(tags_csv).exists():
        raise FileNotFoundError(f"Tags CSV file not found: {tags_csv}")
    tags_df = pd.read_csv(tags_csv)
    if 'name' not in tags_df.columns:
        raise ValueError(f"{tags_csv} must contain 'name' column")
    label_to_idx = {label: idx for idx, label in enumerate(tags_df["name"].astype(str).tolist())}

    # 支持多种图像格式并忽略大小写
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(dataset_folder.glob(ext.lower()))
        image_files.extend(dataset_folder.glob(ext.upper()))
    if not image_files:
        raise ValueError(f"No image files found in {dataset_folder}")

    # 收集所有标签并按逗号分割
    skipped = 0
    all_labels = set()
    for img_file in image_files:
        txt_file = img_file.with_suffix(".txt")
        if not txt_file.exists():
            skipped += 1
            logging.warning(f"Skipping {img_file} due to missing .txt file")
            continue
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # 严格按逗号分割标签
                labels = [label.strip() for label in content.split(",") if label.strip()]
                if not labels:
                    logging.warning(f"Empty or invalid label content in {txt_file}: '{content}'")
                    skipped += 1
                    continue
                all_labels.update(labels)
        except Exception as e:
            logging.warning(f"Failed to read {txt_file}: {e}")
            skipped += 1
            continue

    if skipped > 0:
        logging.info(f"Skipped {skipped} images due to missing or invalid label files")

    # 更新 tags_csv 文件，检查并添加新标签
    tags_df = update_tags_csv(tags_csv, all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(tags_df["name"].astype(str).tolist())}

    # 划分训练集和验证集
    np.random.seed(42)
    indices = np.random.permutation(len(image_files))
    train_size = int(len(image_files) * split_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 创建训练集 TFRecord
    train_dir = output_path / "record_shards_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(train_dir / "aibooru_train.tfrecord")) as train_writer:
        for idx in tqdm(train_indices, desc="Creating train TFRecord"):
            img_file = image_files[idx]
            txt_file = img_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    labels = [label.strip() for label in content.split(",") if label.strip()]
                label_indexes = [label_to_idx[label] for label in labels if label in label_to_idx]
                example = create_example(str(img_file), label_indexes, img_size)
                if example is not None:
                    train_writer.write(example.SerializeToString())
            except Exception as e:
                logging.warning(f"Failed to process {img_file}: {e}")

    # 创建验证集 TFRecord
    val_dir = output_path / "record_shards_val"
    val_dir.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(val_dir / "aibooru_val.tfrecord")) as val_writer:
        for idx in tqdm(val_indices, desc="Creating val TFRecord"):
            img_file = image_files[idx]
            txt_file = img_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    labels = [label.strip() for label in content.split(",") if label.strip()]
                label_indexes = [label_to_idx[label] for label in labels if label in label_to_idx]
                example = create_example(str(img_file), label_indexes, img_size)
                if example is not None:
                    val_writer.write(example.SerializeToString())
            except Exception as e:
                logging.warning(f"Failed to process {img_file}: {e}")

    # 保存元数据
    meta = {
        "num_classes": len(label_to_idx),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
    }
    with open(output_path / "aibooru.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"TFRecords created successfully at {output_path}")
    logging.info(f"Number of classes: {meta['num_classes']}")
    logging.info(f"Train samples: {meta['train_samples']}")
    logging.info(f"Val samples: {meta['val_samples']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TFRecord files from image dataset")
    parser.add_argument("--dataset-folder", type=str, required=True, help="Folder containing images and labels")
    parser.add_argument("--output-path", type=str, required=True, help="Output path for TFRecord files")
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Train/validation split ratio")
    parser.add_argument("--img-size", type=int, default=224, help="Image size for resizing")
    parser.add_argument("--tags-csv", type=str, default="selected_tags.csv", help="Path to selected_tags.csv")
    
    args = parser.parse_args()
    create_tfrecords(args.dataset_folder, args.output_path, args.split_ratio, args.img_size, args.tags_csv)