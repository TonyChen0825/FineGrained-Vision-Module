# configs/paths.py
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===== 项目内目录 =====
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
SAMPLES_IMAGES_DIR = os.path.join(SAMPLES_DIR, "images")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


SAM_HQ_WEIGHT = os.path.join(MODELS_DIR, "sam-hq", "sam_hq_vit_b.pth")

# ===== 原始数据路径=====

RAW_DATA_ROOT = "C:/Users/29941/Desktop/albb/data" 
DEEPFASHION2_ROOT = os.path.join(RAW_DATA_ROOT, "deepfashion2")

# 确保输出目录存在
os.makedirs(SAMPLES_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)