# src/utils/metrics.py
import numpy as np
from PIL import Image, ImageDraw


def parse_polygon_to_mask(polygon, width, height):
    """
    将 DeepFashion2 的 segmentation 多边形转为二值掩码
    polygon: 多边形点列表，如 [[x1,y1, x2,y2, ...]]
    width: 图片宽度
    height: 图片高度
    返回: (height, width) 的二值掩码 (0/255)
    """
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    if isinstance(polygon, list):
        if polygon and isinstance(polygon[0], (list, tuple)):
            for poly in polygon:
                if len(poly) >= 6:
                    draw.polygon(poly, fill=255)
        else:
            if len(polygon) >= 6:
                draw.polygon(polygon, fill=255)
    
    return np.array(mask)


def compute_iou(mask1, mask2):
    """
    计算两个二值掩码的 IoU（交并比）
    mask1, mask2: 二值掩码，像素值为 0 或 255
    返回: IoU 值 (0~1)
    """
    m1 = mask1 > 0
    m2 = mask2 > 0
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def compute_batch_iou(pred_masks, gt_masks):
    """
    批量计算 IoU
    pred_masks: 预测掩码列表
    gt_masks: 真实掩码列表
    返回: IoU 列表和平均 IoU
    """
    ious = []
    for pred, gt in zip(pred_masks, gt_masks):
        if pred is not None and gt is not None:
            iou = compute_iou(pred, gt)
            ious.append(iou)
    if ious:
        return ious, np.mean(ious)
    return [], 0.0