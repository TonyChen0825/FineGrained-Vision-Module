# scripts/run_segmentation.py
import sys
import os
import json
import time
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.paths import (SAMPLES_DIR, SAMPLES_IMAGES_DIR, RESULTS_DIR, LIGHT_HQSAM_WEIGHT)
from src.models.light_hqsam import LightHQSAMSegmentor
from src.utils.metrics import parse_polygon_to_mask, compute_iou


def main():
    # 检查模型权重
    if not os.path.exists(LIGHT_HQSAM_WEIGHT):
        print(f" 找不到模型权重: {LIGHT_HQSAM_WEIGHT}")
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
    segmentor = LightHQSAMSegmentor(LIGHT_HQSAM_WEIGHT)
    
    # 初始化 IoU 记录
    all_ious = []
    all_iou_per_category = {}
    
    # 逐张处理
    total_time = 0
    total_objects = 0
    results_summary = []
    
    print("\n 开始分割（含 IoU 评估）...")
    
    for sample in tqdm(samples, desc="处理进度"):
        img_path = os.path.join(SAMPLES_IMAGES_DIR, sample['file_name'])
        
        if not os.path.exists(img_path):
            print(f"   跳过: {sample['file_name']} 不存在")
            continue
        
        # 提取 bbox、categories 和 gt_masks
        bboxes = []
        categories = []
        gt_masks = []
        
        from PIL import Image
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        for ann in sample['annotations']:
            b = ann['bbox']
            bboxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            categories.append(ann.get('category_name', '未知'))
            
            # 提取真实掩码
            if 'segmentation' in ann and ann['segmentation']:
                try:
                    gt_mask = parse_polygon_to_mask(
                        ann['segmentation'],
                        img_width,
                        img_height
                    )
                    gt_masks.append(gt_mask)
                except Exception as e:
                    gt_masks.append(None)
            else:
                gt_masks.append(None)
        
        if not bboxes:
            continue
        
        # 推理
        start = time.time()
        try:
            results = segmentor.predict(img_path, bboxes)
            elapsed = time.time() - start
            total_time += elapsed
            total_objects += len(results)
            
            # 计算 IoU
            for i, res in enumerate(results):
                pred_mask = res['mask']
                gt_mask = gt_masks[i] if i < len(gt_masks) else None
                cat = categories[i] if i < len(categories) else '未知'
                
                if gt_mask is not None and pred_mask is not None:
                    iou = compute_iou(pred_mask, gt_mask)
                    all_ious.append(iou)
                    
                    if cat not in all_iou_per_category:
                        all_iou_per_category[cat] = []
                    all_iou_per_category[cat].append(iou)
            
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
    
    # IoU 统计
    if all_ious:
        print("\n" + "="*50)
        print(" IoU 评估结果")
        print("="*50)
        print(f"  评估对象数: {len(all_ious)}")
        print(f"  平均 IoU: {np.mean(all_ious):.4f}")
        print(f"  最小 IoU: {np.min(all_ious):.4f}")
        print(f"  最大 IoU: {np.max(all_ious):.4f}")
        print(f"  中位数 IoU: {np.median(all_ious):.4f}")
        
        if all_iou_per_category:
            print("\n  各类别 IoU:")
            for cat, ious in sorted(all_iou_per_category.items(), key=lambda x: np.mean(x[1]), reverse=True):
                print(f"    {cat}: {np.mean(ious):.4f} (n={len(ious)})")
        
        avg_iou = np.mean(all_ious)
        if avg_iou >= 0.85:
            print(f"\n   平均 IoU = {avg_iou:.4f} (已达到需求文档要求 ≥0.85)")
        else:
            print(f"\n   平均 IoU = {avg_iou:.4f} (未达到需求文档要求 ≥0.85，差 {0.85 - avg_iou:.4f})")
    
    # 类别统计
    cat_count = {}
    for s in results_summary:
        for cat in s['categories']:
            cat_count[cat] = cat_count.get(cat, 0) + 1
    
    print("\n  各类别数量:")
    for name, count in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count}")
    
    # 保存报告
    report = {
        'total_images': len(results_summary),
        'total_objects': total_objects,
        'total_time': total_time,
        'avg_time_per_image_ms': total_time / len(results_summary) * 1000 if results_summary else 0,
        'avg_time_per_object_ms': total_time / total_objects * 1000 if total_objects else 0,
        'category_distribution': cat_count,
        'iou_avg': np.mean(all_ious) if all_ious else None,
        'iou_min': np.min(all_ious) if all_ious else None,
        'iou_max': np.max(all_ious) if all_ious else None,
        'iou_per_category': {cat: np.mean(ious) for cat, ious in all_iou_per_category.items()},
        'meets_requirement': np.mean(all_ious) >= 0.85 if all_ious else False
    }
    
    report_path = os.path.join(RESULTS_DIR, "report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n 报告已保存: {report_path}")
    print(f" 结果图片保存在: {RESULTS_DIR}")


if __name__ == "__main__":
    main()