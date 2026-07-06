# scripts/run_segmentation.py
import sys
import os
import json
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import (
    SAMPLES_DIR, SAMPLES_IMAGES_DIR, RESULTS_DIR, SAM_HQ_WEIGHT
)
from src.models.sam_hq import SAMHQSgmentor


def main():
    # 检查模型权重
    if not os.path.exists(SAM_HQ_WEIGHT):
        print(f" 找不到模型权重: {SAM_HQ_WEIGHT}")
        return
    
    # 加载标注
    ann_file = os.path.join(SAMPLES_DIR, "annotations.json")
    if not os.path.exists(ann_file):
        print(f" 找不到标注文件: {ann_file}")
        return
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f" 加载 {len(samples)} 张图片的标注")
    
    # 初始化分割器
    segmentor = SAMHQSgmentor(SAM_HQ_WEIGHT)
    
    # 逐张处理
    total_time = 0
    total_objects = 0
    results_summary = []
    
    print("\n 开始分割...")
    
    for sample in tqdm(samples, desc="处理进度"):
        img_path = os.path.join(SAMPLES_IMAGES_DIR, sample['file_name'])
        
        if not os.path.exists(img_path):
            print(f"   跳过: {sample['file_name']} 不存在")
            continue
        
        # 提取bbox和类别
        bboxes = []
        categories = []
        for ann in sample['annotations']:
            bbox = ann['bbox']
            # 确保是 [x1, y1, x2, y2] 格式
            if len(bbox) == 4:
                bboxes.append(bbox)
                categories.append(ann.get('category_name', '未知'))
        
        if not bboxes:
            continue
        
        # 推理
        start = time.time()
        try:
            results = segmentor.predict(img_path, bboxes)
            elapsed = time.time() - start
            total_time += elapsed
            total_objects += len(results)
            
            # 保存可视化结果
            save_path = os.path.join(RESULTS_DIR, f"{sample['image_id']}_result.jpg")
            segmentor.visualize(img_path, results, save_path, categories)
            
            results_summary.append({
                'image_id': sample['image_id'],
                'file_name': sample['file_name'],
                'num_objects': len(results),
                'time': elapsed,
                'categories': categories
            })
            
        except Exception as e:
            print(f"   处理 {sample['file_name']} 失败: {e}")
    
    # 统计报告
    print("\n" + "="*50)
    print(" 分割完成统计")
    print("="*50)
    print(f"  处理图片数: {len(results_summary)}")
    print(f"  分割对象数: {total_objects}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均每张: {total_time/len(results_summary)*1000:.2f}ms")
    if total_objects > 0:
        print(f"  平均每个对象: {total_time/total_objects*1000:.2f}ms")
    
    cat_count = {}
    for s in results_summary:
        for cat in s['categories']:
            cat_count[cat] = cat_count.get(cat, 0) + 1
    
    print("\n  各类别数量:")
    for name, count in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}")
    
    report = {
        'total_images': len(results_summary),
        'total_objects': total_objects,
        'total_time': total_time,
        'avg_time_per_image_ms': total_time / len(results_summary) * 1000 if results_summary else 0,
        'category_distribution': cat_count,
    }
    
    report_path = os.path.join(RESULTS_DIR, "report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n 报告已保存: {report_path}")
    print(f" 结果图片保存在: {RESULTS_DIR}")


if __name__ == "__main__":
    main()