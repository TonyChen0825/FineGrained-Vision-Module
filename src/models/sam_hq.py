# src/models/sam_hq.py
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 尝试导入 SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print(" segment-anything 未安装，使用简化模式")


class SAMHQSgmentor:
    def __init__(self, weight_path, model_type="vit_b"):
        """
        初始化分割器
        weight_path: .pth权重文件路径
        model_type: vit_b / vit_l / vit_h
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weight_path = weight_path
        
        print(f" 加载SAM...")
        print(f"  权重: {weight_path}")
        print(f"  设备: {self.device}")
        
        if SAM_AVAILABLE:
            # 用官方SAM加载
            self.model = sam_model_registry[model_type](checkpoint=weight_path)
            self.model.to(device=self.device)
            self.model.eval()
            self.predictor = SamPredictor(self.model)
            print(" SAM加载完成（官方模式）")
        else:
            # 简化模式
            self.model = None
            self.predictor = None
            print("⚠️ 使用简化模式（仅显示bbox）")
    
    def predict(self, image_path, bboxes):
        """对一张图片中的多个bbox进行分割"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        results = []
        
        if SAM_AVAILABLE and self.predictor:
            # 用SAM分割
            self.predictor.set_image(image_rgb)
            
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                input_box = np.array([x1, y1, x2, y2])
                masks, scores, _ = self.predictor.predict(
                    box=input_box,
                    multimask_output=False
                )
                mask = (masks[0] > 0.5).astype(np.uint8) * 255
                results.append({
                    'mask': mask,
                    'bbox': [x1, y1, x2, y2],
                    'area': int(np.sum(mask > 0))
                })
        else:
            # 简化模式：直接用bbox作为掩码
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                results.append({
                    'mask': mask,
                    'bbox': [x1, y1, x2, y2],
                    'area': (y2-y1) * (x2-x1)
                })
        
        return results
    
    def visualize(self, image_path, results, save_path=None, categories=None):
        """可视化分割结果"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 128, 0), (128, 0, 128)
        ]
        
        for i, res in enumerate(results):
            color = colors[i % len(colors)]
            mask = res['mask']
            bbox = res['bbox']
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = categories[i] if categories and i < len(categories) else f"#{i+1}"
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"   保存: {save_path}")
        
        return image