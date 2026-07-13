# scripts/run_localization.py
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import SAMPLES_DIR, SAMPLES_IMAGES_DIR
from src.models.language_guided_localizer import LanguageGuidedLocalizer


def main():
    # 加载样本
    ann_file = os.path.join(SAMPLES_DIR, "annotations.json")
    if not os.path.exists(ann_file):
        print(f" 找不到: {ann_file}")
        print("请先运行 python scripts/extract_samples.py")
        return
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f" 加载 {len(samples)} 张图片")
    
    # 初始化定位器
    localizer = LanguageGuidedLocalizer(
        text_encoder_type="sentence-transformers"
    )
    
    # 测试查询
    test_queries = ["领口", "袖口", "图案", "腰部", "下摆"]
    
    for i, sample in enumerate(samples):
        image_path = os.path.join(SAMPLES_IMAGES_DIR, sample['file_name'])
        if not os.path.exists(image_path):
            continue
        
        regions = []
        categories = []
        for ann in sample['annotations']:
            b = ann['bbox']
            regions.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            categories.append(ann.get('category_name', '未知'))
        
        print(f"\n 图片 {i+1}: {sample['file_name']}")
        print(f"   候选区域: {len(regions)} 个 ({categories})")
        
        for query in test_queries:
            result = localizer.localize(image_path, query, regions)
            if result:
                print(f"    '{query}' → 区域 {result['best_region_idx']} (置信度: {result['confidence']:.3f})")
            else:
                print(f"    '{query}' → 失败")


if __name__ == "__main__":
    main()