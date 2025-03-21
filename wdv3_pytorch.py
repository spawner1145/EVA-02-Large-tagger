import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import os

from models import EVA02Transformer, eva02_large

MODEL_REPO_MAP = {
    "eva02_large": "SmilingWolf/wd-eva02-large-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "vit_large": "SmilingWolf/wd-vit-large-tagger-v3",
    "swinv2_v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "swinv2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = (
            image.convert("RGBA")
            if "transparency" in image.info
            else image.convert("RGB")
        )
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

class LabelData:
    def __init__(self, names: List[str], rating: List[int], general: List[int], character: List[int], artist: List[int]):
        self.names = names
        self.rating = rating
        self.general = general
        self.character = character
        self.artist = artist

def load_labels_local(csv_path: str) -> LabelData:
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    if not {"name", "category"}.issubset(df.columns):
        raise ValueError(f"CSV file {csv_path} must contain 'name' and 'category' columns")
    df = df.dropna(subset=['name']).fillna({'category': 0})
    tag_data = LabelData(
        names=df["name"].astype(str).tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
        artist=list(np.where(df["category"] == 5)[0]),
    )
    return tag_data

def load_model_local(model_path: str, config_path: str, num_classes: int) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config path not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_builder = eva02_large()
    model = model_builder.build(num_classes=num_classes)
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    model.eval()
    return model, config["image_size"]

def load_thresholds(threshold_file: str) -> Dict[str, float]:
    """从 threshold.json 文件加载阈值，若文件不存在则使用默认值"""
    default_thresholds = {
        "general_threshold": 0.35,
        "character_threshold": 0.75,
        "artist_threshold": 0.75,
        "caption_threshold": 0.35,
        "tags_threshold": 0.35
    }
    threshold_path = Path(threshold_file)
    if threshold_path.exists():
        try:
            with open(threshold_path, 'r') as f:
                thresholds = json.load(f)
            # 确保所有必要的键存在，若缺少则使用默认值
            for key in default_thresholds:
                if key not in thresholds:
                    thresholds[key] = default_thresholds[key]
            return thresholds
        except Exception as e:
            print(f"Failed to load {threshold_file}: {e}, using default thresholds")
            return default_thresholds
    else:
        print(f"{threshold_file} not found, using default thresholds")
        return default_thresholds

def get_tags(probs: torch.Tensor, labels: LabelData, thresholds: Dict[str, float]) -> tuple:
    probs = probs.cpu().numpy()
    if len(probs) != len(labels.names):
        raise ValueError(f"Probs length {len(probs)} does not match labels length {len(labels.names)}")
    probs = list(zip(labels.names, probs))

    # Rating 标签（通常不需要阈值，直接输出最高概率）
    rating_labels = dict([probs[i] for i in labels.rating])

    # General 标签
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > thresholds["general_threshold"]])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character 标签
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > thresholds["character_threshold"]])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Artist 标签
    artist_labels = [probs[i] for i in labels.artist]
    artist_labels = dict([x for x in artist_labels if x[1] > thresholds["artist_threshold"]])
    artist_labels = dict(sorted(artist_labels.items(), key=lambda item: item[1], reverse=True))

    # Caption 和 Tags（基于所有类别标签的组合）
    combined_names = []
    for label, prob in gen_labels.items():
        if prob > thresholds["caption_threshold"]:
            combined_names.append(label)
    for label, prob in char_labels.items():
        if prob > thresholds["caption_threshold"]:
            combined_names.append(label)
    for label, prob in artist_labels.items():
        if prob > thresholds["caption_threshold"]:
            combined_names.append(label)
    caption = ", ".join(combined_names)

    # Tags 使用 tags_threshold 筛选
    tag_names = []
    for label, prob in gen_labels.items():
        if prob > thresholds["tags_threshold"]:
            tag_names.append(label)
    for label, prob in char_labels.items():
        if prob > thresholds["tags_threshold"]:
            tag_names.append(label)
    for label, prob in artist_labels.items():
        if prob > thresholds["tags_threshold"]:
            tag_names.append(label)
    taglist = ", ".join(tag_names).replace("_", " ").replace("(", r"\(").replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels, artist_labels

def main():
    parser = argparse.ArgumentParser(description="Inference script for tagging images")
    parser.add_argument("--image_file", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="eva02_large", help="Model name")
    parser.add_argument("--gen_threshold", type=float, help="Threshold for general tags (overrides threshold.json)")
    parser.add_argument("--char_threshold", type=float, help="Threshold for character tags (overrides threshold.json)")
    parser.add_argument("--artist_threshold", type=float, help="Threshold for artist tags (overrides threshold.json)")
    parser.add_argument("--caption_threshold", type=float, help="Threshold for caption (overrides threshold.json)")
    parser.add_argument("--tags_threshold", type=float, help="Threshold for tags (overrides threshold.json)")
    parser.add_argument("--model_path", type=str, default="eva02_large.pth", help="Path to local model weights")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to model config")
    parser.add_argument("--tags_csv", type=str, default="selected_tags.csv", help="Path to local selected_tags.csv")
    parser.add_argument("--threshold_json", type=str, default="threshold.json", help="Path to threshold.json")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 threshold.json
    thresholds = load_thresholds(args.threshold_json)

    # 如果命令行提供了阈值，则覆盖 threshold.json 中的值
    if args.gen_threshold is not None:
        thresholds["general_threshold"] = args.gen_threshold
    if args.char_threshold is not None:
        thresholds["character_threshold"] = args.char_threshold
    if args.artist_threshold is not None:
        thresholds["artist_threshold"] = args.artist_threshold
    if args.caption_threshold is not None:
        thresholds["caption_threshold"] = args.caption_threshold
    if args.tags_threshold is not None:
        thresholds["tags_threshold"] = args.tags_threshold

    repo_id = MODEL_REPO_MAP.get(args.model)
    if not repo_id:
        raise ValueError(f"Unknown model name '{args.model}'. Available models: {list(MODEL_REPO_MAP.keys())}")

    image_path = Path(args.image_file).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading model '{args.model}' from local path '{args.model_path}'...")
    labels = load_labels_local(args.tags_csv)
    num_classes = len(labels.names)
    model, target_size = load_model_local(args.model_path, args.config_path, num_classes)
    model = model.to(device)

    print("Loading image and preprocessing...")
    try:
        img_input = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")
    img_input = pil_ensure_rgb(img_input)
    img_input = pil_pad_square(img_input)
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    inputs = transform(img_input).unsqueeze(0).to(device)
    if inputs.shape != (1, 3, target_size, target_size):
        raise ValueError(f"Expected input shape [1, 3, {target_size}, {target_size}], got {inputs.shape}")

    print("Running inference...")
    with torch.no_grad():
        outputs = model(inputs)
        if outputs.shape[1] != num_classes:
            raise ValueError(f"Expected output shape [1, {num_classes}], got {outputs.shape}")
        outputs = torch.sigmoid(outputs).squeeze(0)

    print("Processing results...")
    caption, taglist, ratings, character, general, artist = get_tags(
        probs=outputs,
        labels=labels,
        thresholds=thresholds,
    )

    print("--------")
    print(f"Caption: {caption}")
    print("--------")
    print(f"Tags: {taglist}")
    print("--------")
    print("Ratings:")
    for k, v in ratings.items():
        print(f"  {k}: {v:.3f}")
    print("--------")
    print(f"Character tags (threshold={thresholds['character_threshold']}):")
    for k, v in character.items():
        print(f"  {k}: {v:.3f}")
    print("--------")
    print(f"General tags (threshold={thresholds['general_threshold']}):")
    for k, v in general.items():
        print(f"  {k}: {v:.3f}")
    print("--------")
    print(f"Artist tags (threshold={thresholds['artist_threshold']}):")
    for k, v in artist.items():
        print(f"  {k}: {v:.3f}")
    print("Done!")

if __name__ == "__main__":
    main()