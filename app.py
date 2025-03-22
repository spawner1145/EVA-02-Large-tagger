import gradio as gr
import os
from pathlib import Path
import subprocess
import pandas as pd
import json
import shutil
import sys
import threading
import queue

required_scripts = [
    "simple-dan-crawl.py",
    "generate_selected_tags.py",
    "create_tfrecord.py",
    "replace_blanks.py",
    "train_pytorch.py",
    "wdv3_pytorch.py"
]

for script in required_scripts:
    if not os.path.exists(script):
        raise FileNotFoundError(f"Required script {script} not found in the current directory.")

def normalize_path(path_str):
    if not path_str:
        return None
    path_str = path_str.strip()
    if path_str.startswith("./"):
        path_str = path_str[2:]
    path = Path(path_str).resolve()
    return path

def read_stream(stream, q, label):
    while True:
        line = stream.readline()
        if not line:
            break
        print(f"[{label}] {line}", end='', file=sys.stdout if label == "STDOUT" else sys.stderr)
        q.put((label, line))

def run_command_streaming(cmd):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        errors='replace',
        env=env
    )
    
    q = queue.Queue()
    
    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, q, "STDOUT"))
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, q, "STDERR"))
    stdout_thread.start()
    stderr_thread.start()
    
    output = ""
    while True:
        try:
            label, line = q.get(timeout=1)
            output += line
            yield output
        except queue.Empty:
            if process.poll() is not None and q.empty():
                break
    
    stdout_thread.join()
    stderr_thread.join()
    
    process.stdout.close()
    process.stderr.close()
    process.wait()

# Tab 1: 数据下载
def download_data(target_count, output_dir):
    output_dir = normalize_path(output_dir)
    if output_dir is None:
        yield "Error: Output directory cannot be empty."
        return
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    with open("simple-dan-crawl.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("target_count = 10", f"target_count = {target_count}")
    content = content.replace("os.path.exists('dataset')", f"os.path.exists('{output_dir}')")
    content = content.replace("os.path.join('dataset', filename)", f"os.path.join('{output_dir}', filename)")
    content = content.replace("os.path.join('dataset', txt_filename)", f"os.path.join('{output_dir}', txt_filename)")
    
    with open("simple-dan-crawl.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    cmd = ["python", "-u", "simple-dan-crawl.py"]
    for output in run_command_streaming(cmd):
        yield output
    
    content = content.replace(f"target_count = {target_count}", "target_count = 10")
    content = content.replace(f"os.path.exists('{output_dir}')", "os.path.exists('dataset')")
    content = content.replace(f"os.path.join('{output_dir}', filename)", "os.path.join('dataset', filename)")
    content = content.replace(f"os.path.join('{output_dir}', txt_filename)", "os.path.join('dataset', txt_filename)")
    
    with open("simple-dan-crawl.py", "w", encoding="utf-8") as f:
        f.write(content)

# Tab 2: 数据预处理 - 生成 selected_tags.csv
def generate_tags(dataset_folder, output_csv):
    dataset_folder = normalize_path(dataset_folder)
    output_csv = normalize_path(output_csv)
    
    if dataset_folder is None or output_csv is None:
        yield "Error: Dataset folder and output CSV path cannot be empty."
        return
    
    if not dataset_folder.exists():
        yield f"Error: Dataset folder {dataset_folder} does not exist."
        return
    
    cmd = [
        "python", "-u", "generate_selected_tags.py",
        "--dataset-folder", str(dataset_folder),
        "--output-csv", str(output_csv)
    ]
    for output in run_command_streaming(cmd):
        yield output

# Tab 2: 数据预处理 - 转换为 TFRecord
def create_tfrecord(dataset_folder, output_path, split_ratio, img_size, tags_csv):
    dataset_folder = normalize_path(dataset_folder)
    output_path = normalize_path(output_path)
    tags_csv = normalize_path(tags_csv)
    
    if dataset_folder is None or output_path is None or tags_csv is None:
        yield "Error: Dataset folder, output path, and tags CSV cannot be empty."
        return
    
    if not dataset_folder.exists():
        yield f"Error: Dataset folder {dataset_folder} does not exist."
        return
    if not tags_csv.exists():
        yield f"Error: Tags CSV file {tags_csv} does not exist."
        return
    
    cmd = [
        "python", "-u", "create_tfrecord.py",
        "--dataset-folder", str(dataset_folder),
        "--output-path", str(output_path),
        "--split-ratio", str(split_ratio),
        "--img-size", str(img_size),
        "--tags-csv", str(tags_csv)
    ]
    for output in run_command_streaming(cmd):
        yield output

# Tab 2: 数据预处理 - 清理 txt 文件
def replace_blanks(dataset_folder):
    dataset_folder = normalize_path(dataset_folder)
    
    if dataset_folder is None:
        yield "Error: Dataset folder cannot be empty."
        return
    
    if not dataset_folder.exists():
        yield f"Error: Dataset folder {dataset_folder} does not exist."
        return
    
    with open("replace_blanks.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("target_directory = 'dataset'", f"target_directory = '{dataset_folder}'")
    
    with open("replace_blanks.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    cmd = ["python", "-u", "replace_blanks.py"]
    for output in run_command_streaming(cmd):
        yield output
    
    content = content.replace(f"target_directory = '{dataset_folder}'", "target_directory = 'dataset'")
    with open("replace_blanks.py", "w", encoding="utf-8") as f:
        f.write(content)

# Tab 3: 模型训练
def train_model(dataset_root, output_dir, image_size, batch_size, epochs, lr):
    dataset_root = normalize_path(dataset_root)
    output_dir = normalize_path(output_dir)
    
    if dataset_root is None or output_dir is None:
        yield "Error: Dataset root and output directory cannot be empty."
        return
    
    if not dataset_root.exists():
        yield f"Error: Dataset root {dataset_root} does not exist."
        return
    
    cmd = [
        "python", "-u", "train_pytorch.py",
        "--dataset_root", str(dataset_root),
        "--output_dir", str(output_dir),
        "--image_size", str(image_size),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr)
    ]
    for output in run_command_streaming(cmd):
        yield output

# Tab 4: 推理
def run_inference(image_file, model_path, config_path, tags_csv, threshold_json, gen_threshold, char_threshold, artist_threshold, caption_threshold, tags_threshold):
    if image_file is None:
        yield "Error: Please upload an image file."
        return
    image_file_path = Path(image_file.name).resolve()
    
    model_path = normalize_path(model_path)
    config_path = normalize_path(config_path)
    tags_csv = normalize_path(tags_csv)
    threshold_json = normalize_path(threshold_json) if threshold_json else "threshold.json"
    
    if model_path is None or config_path is None or tags_csv is None:
        yield "Error: Model path, config path, and tags CSV cannot be empty."
        return
    
    if not image_file_path.exists():
        yield f"Error: Image file {image_file_path} does not exist."
        return
    if not model_path.exists():
        yield f"Error: Model path {model_path} does not exist."
        return
    if not config_path.exists():
        yield f"Error: Config path {config_path} does not exist."
        return
    if not tags_csv.exists():
        yield f"Error: Tags CSV file {tags_csv} does not exist."
        return
    
    cmd = [
        "python", "-u", "wdv3_pytorch.py",
        "--image_file", str(image_file_path),
        "--model_path", str(model_path),
        "--config_path", str(config_path),
        "--tags_csv", str(tags_csv),
        "--threshold_json", str(threshold_json)
    ]
    
    if gen_threshold is not None and gen_threshold != 0:
        cmd.extend(["--gen_threshold", str(gen_threshold)])
    if char_threshold is not None and char_threshold != 0:
        cmd.extend(["--char_threshold", str(char_threshold)])
    if artist_threshold is not None and artist_threshold != 0:
        cmd.extend(["--artist_threshold", str(artist_threshold)])
    if caption_threshold is not None and caption_threshold != 0:
        cmd.extend(["--caption_threshold", str(caption_threshold)])
    if tags_threshold is not None and tags_threshold != 0:
        cmd.extend(["--tags_threshold", str(tags_threshold)])
    
    for output in run_command_streaming(cmd):
        yield output

# Tab 5: 查看和编辑 selected_tags.csv
def view_tags_csv(tags_csv_path):
    tags_csv_path = normalize_path(tags_csv_path)
    
    if tags_csv_path is None:
        return None, "Error: Tags CSV path cannot be empty."
    
    if not tags_csv_path.exists():
        return None, f"Error: Tags CSV file {tags_csv_path} does not exist."
    
    df = pd.read_csv(tags_csv_path)
    return df, f"Loaded {tags_csv_path} successfully."

def save_tags_csv(tags_csv_path, edited_df):
    tags_csv_path = normalize_path(tags_csv_path)
    
    if tags_csv_path is None:
        return "Error: Tags CSV path cannot be empty."
    
    if edited_df is None:
        return "Error: No data to save."
    
    try:
        edited_df.to_csv(tags_csv_path, index=False)
        return f"Successfully saved to {tags_csv_path}."
    except Exception as e:
        return f"Error saving file: {str(e)}"

# Tab 6: 查看和编辑 threshold.json
def view_threshold_json(threshold_json_path):
    threshold_json_path = normalize_path(threshold_json_path)
    
    if threshold_json_path is None:
        return None, "Error: Threshold JSON path cannot be empty."
    
    if not threshold_json_path.exists():
        return None, f"Error: Threshold JSON file {threshold_json_path} does not exist."
    
    with open(threshold_json_path, "r") as f:
        data = json.load(f)
    return json.dumps(data, indent=2), f"Loaded {threshold_json_path} successfully."

def save_threshold_json(threshold_json_path, edited_json):
    threshold_json_path = normalize_path(threshold_json_path)
    
    if threshold_json_path is None:
        return "Error: Threshold JSON path cannot be empty."
    
    if not edited_json:
        return "Error: No data to save."
    
    try:
        data = json.loads(edited_json)
        with open(threshold_json_path, "w") as f:
            json.dump(data, f, indent=2)
        return f"Successfully saved to {threshold_json_path}."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# EVA-02 Large 图像标签模型 - Gradio 界面")
    gr.Markdown("**注意**：所有路径输入支持相对路径和绝对路径。相对路径请以 `./` 开头（如 `./dataset`），或直接输入文件夹/文件名（如 `dataset`），系统会自动解析为当前工作目录下的路径。")
    
    gr.HTML("""
    <style>
        /* 为所有 gr.Textbox 的 textarea 添加滚动条 */
        .gr-textbox textarea {
            overflow-y: auto !important;
            max-height: 300px !important; /* 限制高度，超出后显示滚动条 */
            min-height: 150px !important; /* 最小高度 */
        }
    </style>
    """)
    
    with gr.Tab("数据下载"):
        gr.Markdown("### 从 Danbooru 下载图像和标签")
        target_count = gr.Number(label="下载图片数量", value=10, precision=0)
        output_dir = gr.Textbox(label="输出目录", value="./dataset", placeholder="请输入相对路径（如 ./dataset）或绝对路径")
        download_btn = gr.Button("开始下载")
        download_output = gr.Textbox(label="输出日志", lines=10, max_lines=10)
        download_btn.click(
            fn=download_data,
            inputs=[target_count, output_dir],
            outputs=download_output
        )
    
    with gr.Tab("数据预处理"):
        gr.Markdown("### 数据预处理步骤")
        
        with gr.Group():
            gr.Markdown("#### 1. 生成 selected_tags.csv（非必须）")
            gen_tags_dataset_folder = gr.Textbox(label="数据集目录", value="./dataset", placeholder="请输入相对路径（如 ./dataset）或绝对路径")
            gen_tags_output_csv = gr.Textbox(label="输出 CSV 路径", value="./selected_tags.csv", placeholder="请输入相对路径（如 ./selected_tags.csv）或绝对路径")
            gen_tags_btn = gr.Button("生成 selected_tags.csv")
            gen_tags_output = gr.Textbox(label="输出日志", lines=10, max_lines=10)
            gen_tags_btn.click(
                fn=generate_tags,
                inputs=[gen_tags_dataset_folder, gen_tags_output_csv],
                outputs=gen_tags_output
            )
            
        with gr.Group():
            gr.Markdown("#### 2. 清理 txt 文件（将空格替换为逗号）（非必须）")
            replace_blanks_folder = gr.Textbox(label="数据集目录", value="./dataset", placeholder="请输入相对路径（如 ./dataset）或绝对路径")
            replace_blanks_btn = gr.Button("清理 txt 文件")
            replace_blanks_output = gr.Textbox(label="输出日志", lines=10, max_lines=10)
            replace_blanks_btn.click(
                fn=replace_blanks,
                inputs=replace_blanks_folder,
                outputs=replace_blanks_output
            )
        
        with gr.Group():
            gr.Markdown("#### 3. 转换为 TFRecord 格式（必须）")
            tfrecord_dataset_folder = gr.Textbox(label="数据集目录", value="./dataset", placeholder="请输入相对路径（如 ./dataset）或绝对路径")
            tfrecord_output_path = gr.Textbox(label="输出路径", value="./output-datasets", placeholder="请输入相对路径（如 ./output-datasets）或绝对路径")
            tfrecord_split_ratio = gr.Slider(label="训练/验证集划分比例", minimum=0.1, maximum=0.9, value=0.7, step=0.1)
            tfrecord_img_size = gr.Number(label="图像尺寸", value=224, precision=0)
            tfrecord_tags_csv = gr.Textbox(label="Tags CSV 路径", value="./selected_tags.csv", placeholder="请输入相对路径（如 ./selected_tags.csv）或绝对路径")
            tfrecord_btn = gr.Button("转换为 TFRecord")
            tfrecord_output = gr.Textbox(label="输出日志", lines=10, max_lines=10)
            tfrecord_btn.click(
                fn=create_tfrecord,
                inputs=[tfrecord_dataset_folder, tfrecord_output_path, tfrecord_split_ratio, tfrecord_img_size, tfrecord_tags_csv],
                outputs=tfrecord_output
            )
    
    with gr.Tab("模型训练"):
        gr.Markdown("### 训练 EVA-02 Large 模型")
        train_dataset_root = gr.Textbox(label="数据集根目录", value="./output-datasets", placeholder="请输入相对路径（如 ./output-datasets）或绝对路径")
        train_output_dir = gr.Textbox(label="检查点保存目录", value="./checkpoints", placeholder="请输入相对路径（如 ./checkpoints）或绝对路径")
        train_image_size = gr.Number(label="图像尺寸", value=224, precision=0)
        train_batch_size = gr.Number(label="批次大小", value=32, precision=0)
        train_epochs = gr.Number(label="训练轮数", value=10, precision=0)
        train_lr = gr.Number(label="学习率", value=1e-4, precision=6)
        train_btn = gr.Button("开始训练")
        train_output = gr.Textbox(label="输出日志", lines=10, max_lines=10)
        train_btn.click(
            fn=train_model,
            inputs=[train_dataset_root, train_output_dir, train_image_size, train_batch_size, train_epochs, train_lr],
            outputs=train_output
        )
    
    with gr.Tab("推理"):
        gr.Markdown("### 使用模型进行图像标签推理")
        inference_image_file = gr.File(label="输入图像", file_types=[".jpg", ".jpeg", ".png", ".webp"])
        inference_model_path = gr.Textbox(label="模型权重路径", value="./checkpoints/eva02_large_best.pth", placeholder="请输入相对路径（如 ./checkpoints/eva02_large_best.pth）或绝对路径")
        inference_config_path = gr.Textbox(label="配置文件路径", value="./checkpoints/config.json", placeholder="请输入相对路径（如 ./checkpoints/config.json）或绝对路径")
        inference_tags_csv = gr.Textbox(label="Tags CSV 路径", value="./selected_tags.csv", placeholder="请输入相对路径（如 ./selected_tags.csv）或绝对路径")
        inference_threshold_json = gr.Textbox(label="阈值文件路径（可选）", value="./threshold.json", placeholder="请输入相对路径（如 ./threshold.json）或绝对路径")
        with gr.Group():
            gr.Markdown("#### 可选：覆盖阈值（输入 0 或留空则使用 threshold.json 中的默认值）")
            inference_gen_threshold = gr.Number(label="通用标签阈值", value=None, precision=2)
            inference_char_threshold = gr.Number(label="角色标签阈值", value=None, precision=2)
            inference_artist_threshold = gr.Number(label="艺术家标签阈值", value=None, precision=2)
            inference_caption_threshold = gr.Number(label="Caption 阈值", value=None, precision=2)
            inference_tags_threshold = gr.Number(label="Tags 阈值", value=None, precision=2)
        inference_btn = gr.Button("运行推理")
        inference_output = gr.Textbox(label="推理结果", lines=10, max_lines=10)
        inference_btn.click(
            fn=run_inference,
            inputs=[
                inference_image_file,
                inference_model_path,
                inference_config_path,
                inference_tags_csv,
                inference_threshold_json,
                inference_gen_threshold,
                inference_char_threshold,
                inference_artist_threshold,
                inference_caption_threshold,
                inference_tags_threshold
            ],
            outputs=inference_output
        )
    
    with gr.Tab("查看和编辑 selected_tags.csv"):
        gr.Markdown("### 查看和编辑 selected_tags.csv 内容")
        view_tags_csv_path = gr.Textbox(label="Tags CSV 路径", value="./selected_tags.csv", placeholder="请输入相对路径（如 ./selected_tags.csv）或绝对路径")
        with gr.Row():
            view_tags_btn = gr.Button("加载")
            save_tags_btn = gr.Button("保存")
        view_tags_output = gr.Dataframe(label="Tags CSV 内容", interactive=True)
        view_tags_message = gr.Textbox(label="消息")
        view_tags_btn.click(
            fn=view_tags_csv,
            inputs=view_tags_csv_path,
            outputs=[view_tags_output, view_tags_message]
        )
        save_tags_btn.click(
            fn=save_tags_csv,
            inputs=[view_tags_csv_path, view_tags_output],
            outputs=view_tags_message
        )
    
    with gr.Tab("查看和编辑 threshold.json"):
        gr.Markdown("### 查看和编辑 threshold.json 内容")
        view_threshold_json_path = gr.Textbox(label="Threshold JSON 路径", value="./threshold.json", placeholder="请输入相对路径（如 ./threshold.json）或绝对路径")
        with gr.Row():
            view_threshold_btn = gr.Button("加载")
            save_threshold_btn = gr.Button("保存")
        view_threshold_output = gr.Textbox(label="Threshold JSON 内容", interactive=True, lines=10)
        view_threshold_message = gr.Textbox(label="消息")
        view_threshold_btn.click(
            fn=view_threshold_json,
            inputs=view_threshold_json_path,
            outputs=[view_threshold_output, view_threshold_message]
        )
        save_threshold_btn.click(
            fn=save_threshold_json,
            inputs=[view_threshold_json_path, view_threshold_output],
            outputs=view_threshold_message
        )

demo.launch()