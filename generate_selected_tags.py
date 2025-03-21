import os
from pathlib import Path
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_selected_tags(dataset_folder, output_csv):
    dataset_folder = Path(dataset_folder)
    all_tags = set()

    txt_files = list(dataset_folder.glob("*.txt"))
    if not txt_files:
        logging.warning(f"No .txt files found in {dataset_folder}")

    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logging.warning(f"Empty file: {txt_file}")
                    continue
                # 明确按逗号分割，并清理每个标签
                tags = [tag.strip() for tag in content.split(",") if tag.strip()]
                all_tags.update(tags)
        except Exception as e:
            logging.warning(f"Failed to read {txt_file}: {e}")

    # 定义默认的标签到类别映射
    tag_to_category = {
        "general": 9, "sensitive": 9, "questionable": 9, "explicit": 9,
        "1girl": 0, "1boy": 0, "whiskey": 0, "eyepatch": 0, "playboy_bunny": 0, "sword": 0,
        "m16a1_(girls'_frontline)": 4, "guts_(berserk)": 4,
        "artist_name_1": 5, "artist_name_2": 5
    }

    output_csv = Path(output_csv)
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            existing_df = existing_df.dropna(subset=['name']).fillna({'category': 0})
            existing_names = set(existing_df['name'].astype(str))
        except Exception as e:
            logging.warning(f"Failed to read {output_csv}: {e}")
            existing_df = pd.DataFrame({"name": [], "category": []})
            existing_names = set()
    else:
        existing_df = pd.DataFrame({"name": [], "category": []})
        existing_names = set()

    # 只添加新标签
    new_names = [tag for tag in all_tags if tag not in existing_names]
    new_categories = [tag_to_category.get(tag, 0) for tag in new_names]
    new_df = pd.DataFrame({"name": new_names, "category": new_categories})
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    if len(combined_df['name']) != len(combined_df['category']):
        raise ValueError("Mismatch between name and category counts in combined_df")

    combined_df.to_csv(output_csv, index=False)
    logging.info(f"Generated {output_csv} with {len(combined_df)} tags")
    logging.info(f"Added new tags: {new_names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate selected tags CSV from dataset")
    parser.add_argument("--dataset-folder", type=str, required=True, help="Folder containing label files")
    parser.add_argument("--output-csv", type=str, default="selected_tags.csv", help="Output CSV file path")

    args = parser.parse_args()
    generate_selected_tags(args.dataset_folder, args.output_csv)