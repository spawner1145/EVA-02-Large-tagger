# EVA-02 Large 图像标签模型

这是一个高性能的基于 PyTorch 和 TensorFlow 的图像标签项目，使用 EVA-02 Large Transformer 模型对图像进行多标签分类。项目包括数据下载、预处理、模型训练和推理功能，支持通过 `threshold.json` 自定义推理时的标签输出阈值。

项目启发自[SmilingWolf · GitHub](https://github.com/SmilingWolf) 这个版本增加了artist栏目，为画师串反推打好框架基础

## 项目功能

1. **数据下载**：从 danbooru 简单爬取图像及其标签。
2. **数据预处理**：将图像和标签转换为 TFRecord 格式，并更新标签 CSV。
3. **模型训练**：使用 EVA-02 Large 模型进行训练，支持早停和检查点保存。
4. **推理**：对单张图像进行标签预测，输出分类标签和置信度，可通过 `threshold.json` 调控输出阈值。

## 环境要求

- **Python**: 3.8+
- **依赖库**：

```
pip install torch torchvision tensorflow pillow pandas numpy tqdm httpx
```

## 目录结构

```
project/
├── create_tfrecord.py        # 数据预处理脚本
├── train_pytorch.py          # 模型训练脚本
├── models.py                 # 模型定义
├── generate_selected_tags.py # 生成初始标签 CSV（可选）
├── wdv3_pytorch.py           # 推理脚本
├── simple-dan-crawl.py       # 简易danbooru爬虫
├── replace_blanks.py         # 替换 txt 文件中空格为逗号（可选）
├── threshold.json            # 推理阈值配置文件
└── README.md                 # 使用说明
```

## 使用步骤

### 1. 数据下载

- 使用 `simple-dan-crawl.py` 从 danbooru 简单爬取图像和标签(标签不会下载角色名和画师名，请注意，只会带有年龄分级的tag和普通描述tag)。
- 示例命令：

* 默认下载 10 张图片到 **dataset/** 目录，可修改脚本中的 **target_count** 参数调整数量。
* 输出示例：

  ```
  dataset/
  ├── abc123.jpg
  ├── abc123.txt  # 内容: "general,1girl,whiskey"
  ├── def456.jpg
  ├── def456.txt  # 内容: "sensitive,1boy,sword"
  ```

### 2. 数据预处理

#### a. （可选）生成初始标签 CSV

* 如果没有初始的 **selected_tags.csv**，运行以下命令生成，但是只能定位为普通类标签，艺术家或角色名需要自己改类别：

 ```
 python generate_selected_tags.py --dataset-folder ./dataset --output-csv ./selected_tags.csv
```
* 输出示例（**selected_tags.csv**）：

  ```
  name,category
  general,9
  1girl,0
  whiskey,0
  ```

### 标签类别及对应的 **category** 值（用于在推理时直观地分类，需要在你的selected_tags.csv中编写）

1. **Rating 标签（评级标签）**
   * **category 值** : **9**
   * **描述** : 表示图像的评级或敏感度，通常用于区分内容的安全性或适宜性。
   * **示例标签** :
   * **general**（普通）
   * **sensitive**（敏感）
   * **questionable**（可疑）
   * **explicit**（明确/成人）
   * **用途** : 在推理时，这些标签通常用于评估图像的整体评级，不受特定阈值筛选，直接输出最高概率的评级。
2. **General 标签（通用标签）**
   * **category 值** : **0**
   * **描述** : 表示描述图像内容的一般性标签，通常是广泛的描述性词汇。
   * **示例标签** :
   * **1girl**（一个女孩）
   * **1boy**（一个男孩）
   * **whiskey**（威士忌）
   * **eyepatch**（眼罩）
   * **sword**（剑）
   * **用途** : 这些标签在推理时受 **general_threshold** 控制，描述图像的主要特征。
3. **Character 标签（角色标签）**
   * **category 值** : **4**
   * **描述** : 表示特定角色或人物的标签，通常与虚构作品中的角色相关。
   * **示例标签** :
   * **m16a1_(girls'_frontline)**（《少女前线》中的 M16A1）
   * **guts_(berserk)**（《剑风传奇》中的加斯）
   * **用途** : 在推理时受 **character_threshold** 控制，用于识别图像中的具体角色。
4. **Artist 标签（艺术家标签）**
   * **category 值** : 1
   * **描述** : 表示图像的创作者或艺术家的标签。
   * **示例标签** :
   * **artist_name_1**
   * **artist_name_2**
   * **用途** : 在推理时受 **artist_threshold** 控制，用于标注图像的作者。（项目自带的selected_tags.csv里含有danbooru2023数据集的所有画师名，所以画师应该是比较全的）

     ***注意：你使用我项目里的爬虫爬的只会有0和9两个，如果想要定义画师和角色标签自己去改或者自己写数据预处理***

#### b. 转换为 TFRecord 格式

* 运行以下命令将数据集转换为 TFRecord 格式，并更新 **selected_tags.csv**：

 ```
 python create_tfrecord.py --dataset-folder ./dataset --output-path ./output-datasets --split-ratio 0.7 --img-size 224
```
* 参数说明：

  * **--dataset-folder**: 数据集路径
  * **--output-path**: 输出 TFRecord 文件路径
  * **--split-ratio**: 训练/验证集划分比例（默认 0.7）
  * **--img-size**: 图像尺寸（默认 224）
* 输出：

  * **./output-datasets/record_shards_train/aibooru_train.tfrecord**
  * **./output-datasets/record_shards_val/aibooru_val.tfrecord**
  * **./output-datasets/aibooru.json**
  * 更新后的 **selected_tags.csv**

#### c. （可选）清理 txt 文件

* 如果 **.txt** 文件中标签存在空格分隔，运行以下命令替换为逗号，这是为了兼容danbooru的下划线当空格格式：

 ```
 python replace_blanks.py
```

### 3. 训练模型

* 运行以下命令训练 EVA-02 Large 模型：

 ```
 python train_pytorch.py --dataset_root ./output-datasets --output_dir ./checkpoints
```
* 参数说明：

  * **--dataset_root**: TFRecord 文件路径
  * **--output_dir**: 检查点保存路径
  * **--image_size**: 图像尺寸（默认 224）
  * **--batch_size**: 批次大小（默认 32）
  * **--epochs**: 训练轮数（默认 10）
  * **--lr**: 学习率（默认 1e-4）
* 输出：

  * **./checkpoints/eva02_large_epoch_X.pth**: 每轮模型权重
  * **./checkpoints/eva02_large_best.pth**: 最佳模型权重
  * **./checkpoints/config.json**: 模型配置文件

### 4. 配置推理阈值

* 在项目根目录下创建 **threshold.json**，用于控制推理时的标签输出阈值：

  ```
  {
    "general_threshold": 0.40,
    "character_threshold": 0.80,
    "artist_threshold": 0.70,
    "caption_threshold": 0.40,
    "tags_threshold": 0.45
  }
  ```
* 字段说明：

  * **general_threshold**: 通用标签阈值
  * **character_threshold**: 角色标签阈值
  * **artist_threshold**: 艺术家标签阈值
  * **caption_threshold**: Caption 输出阈值
  * **tags_threshold**: Tags 输出阈值

### 5. 推理

* 使用训练好的模型对图像进行标签预测：

  ```
  python wdv3_pytorch.py --image_file ./test.png --model_path ./checkpoints/eva02_large_epoch_10.pth --config_path ./checkpoints/config.json --tags_csv ./selected_tags.csv
  ```
* 参数说明：

  * **--image_file**: 输入图像路径
  * **--model_path**: 模型权重路径
  * **--config_path**: 配置文件路径
  * **--tags_csv**: 标签 CSV 路径
  * **--threshold_json**: 阈值文件路径（默认 **threshold.json**）
  * 可选覆盖阈值：
    * **--gen_threshold**: 通用标签阈值
    * **--char_threshold**: 角色标签阈值
    * **--artist_threshold**: 艺术家标签阈值
    * **--caption_threshold**: Caption 阈值
    * **--tags_threshold**: Tags 阈值
* 示例（使用默认 **threshold.json**）：

  ```
  python wdv3_pytorch.py --image_file ./test.png --model_path ./checkpoints/eva02_large_epoch_10.pth --config_path ./checkpoints/config.json --tags_csv ./selected_tags.csv
  ```
* 示例（覆盖部分阈值）：

  ```
  python wdv3_pytorch.py --image_file ./test.png --model_path ./checkpoints/eva02_large_epoch_10.pth --config_path ./checkpoints/config.json --tags_csv ./selected_tags.csv --gen_threshold 0.50 --char_threshold 0.85
  ```
* 输出示例：

  ```
  Caption: 1girl, m16a1_(girls'_frontline), artist_name_1
  Tags: 1girl, m16a1_(girls'_frontline)
  Ratings:
    general: 0.950
  Character tags (threshold=0.80):
    m16a1_(girls'_frontline): 0.880
  General tags (threshold=0.40):
    1girl: 0.920
    whiskey: 0.450
  Artist tags (threshold=0.70):
    artist_name_1: 0.850
  ```

## 注意事项

1. **依赖安装** ：确保所有依赖已安装。
2. **文件路径** ：检查所有输入路径是否正确。
3. **数据一致性** ：

* **.txt** 文件中的标签必须以逗号分隔。
* 图像和标签文件需配对。

1. **硬件要求** ：

* 训练需要 GPU 支持。
* 推理可在 CPU 上运行，但 GPU 会更快。

1. **阈值配置** ：

* 若 **threshold.json** 缺失，默认使用 0.35（通用/标题/标签）和 0.75（角色/艺术家）。
* 命令行参数优先级高于 **threshold.json**。

## 常见问题

* **Q：TFRecord 文件生成失败？**
  * **A** ：检查图像文件是否损坏，**.txt** 文件是否缺失或格式错误。
* **Q：推理时模型加载失败？**
  * **A** ：确保 **--model_path** 和 **--config_path** 指向正确文件。
* **Q：标签未正确输出？**
  * **A** ：检查 **threshold.json** 或命令行阈值设置是否过高。
