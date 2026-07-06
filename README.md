# FineGrained-Vision-Module

细粒度视觉基础模块 —— 服饰属性识别与分割系统

## 项目简介

本项目实现基于 SAM-HQ 的服饰细粒度分割与属性识别能力，支持：
- 8 大类服饰（上衣、裤子、裙子、外套、连衣裙、鞋子、包、配饰）的精准分割
- 基于自然语言描述的局部区域定位（开发中）
- 200+ 种细粒度属性提取（开发中）

## 环境要求

- Python 3.10+
- CUDA 11.8+（推荐使用 GPU）
- 8GB+ GPU 显存（推荐 16GB+）

## 快速开始

### 1. 克隆项目

```bash
git clone <https://github.com/TonyChen0825/FineGrained-Vision-Module>
cd FineGrained-Vision-Module
```

### 2. 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv venv_fashion

# 激活（Windows CMD）
venv_fashion\Scripts\activate.bat

# 激活（Linux/Mac）
source venv_fashion/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 安装 SAM-HQ

```bash
git clone https://github.com/SysCV/sam-hq.git
cd sam-hq
pip install -e .
cd ..
```

### 5. 下载模型权重

将 SAM-HQ 权重文件（`sam_hq_vit_b.pth`）放入 `models/sam-hq/` 目录下。

> 下载地址：[SAM-HQ Model Checkpoints](https://github.com/SysCV/sam-hq#model-checkpoints)

### 6. 准备数据集

从 DeepFashion2 或 FashionAI 数据集中提取样本：

```bash
# 先修改 configs/paths.py 中的 RAW_DATA_ROOT 为你的数据集路径
python scripts/extract_samples.py
```

### 7. 运行服饰分割

```bash
python scripts/run_segmentation.py
```

分割结果将保存在 `outputs/results/` 目录下。

## 项目结构

```
FineGrained-Vision-Module/
├── configs/
│   └── paths.py              # 路径配置文件
├── data/
│   └── samples/              # 样本数据
│       ├── images/           # 样本图片
│       └── annotations.json  # 统一标注文件
├── src/
│   └── models/
│       └── sam_hq.py         # SAM-HQ 分割器封装
├── scripts/
│   ├── extract_samples.py    # 样本提取脚本
│   └── run_segmentation.py   # 分割运行脚本
├── models/
│   └── sam-hq/
│       └── sam_hq_vit_b.pth  # SAM-HQ 模型权重
├── outputs/
│   └── results/              # 分割结果输出
├── requirements.txt          # Python 依赖
└── README.md                 # 项目说明
```

## 当前进展

| 模块 | 状态 | 说明 |
| :--- | :--- | :--- |
| 环境搭建 | ✅ 完成 | Python 虚拟环境 + CUDA + 全部依赖 |
| 数据预处理 | ✅ 完成 | 从 DeepFashion2 提取 100 张样本 |
| 服饰分割 | ✅ 完成 | SAM-HQ 分割 100 张图片，158 个服饰实例 |
| 语言引导定位 | 🚧 开发中 | 基于 DINOv2 + Qwen-VL |
| 属性识别 | 📋 计划中 | 基于 FashionAI 数据集微调 |

## 性能指标

基于 100 张测试样本：

| 指标 | 数值 |
| :--- | :--- |
| 处理图片数 | 100 张 |
| 分割对象数 | 158 件 |
| 总耗时 | 56 秒 |
| 平均每张图片 | 560 ms |
| 平均每个对象 | 355 ms |

## 技术栈

- **PyTorch**：深度学习框架
- **SAM-HQ**：高精度图像分割模型
- **OpenCV**：图像处理
- **Transformers**：预训练模型加载

