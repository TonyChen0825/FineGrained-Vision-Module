# src/models/light_hqsam.py
import torch
import cv2
import numpy as np
from PIL import Image
from segment_anything_hq import sam_model_registry, SamPredictor

class LightHQSAMSegmentor:
    def __init__(self, weight_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f" 加载 Light HQ-SAM...")
        print(f"  权重: {weight_path}")
        print(f"  设备: {self.device}")
        # 使用 'vit_tiny' 加载 Light HQ-SAM
        self.model = sam_model_registry["vit_tiny"](checkpoint=weight_path)
        self.model.to(device=self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)
        print(" Light HQ-SAM 加载完成")

    def predict(self, image_path, bboxes):
        # 读取图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        results = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            input_box = np.array([x1, y1, x2, y2])
            # 进行预测，并启用 HQ 模式
            masks, scores, _ = self.predictor.predict(
                box=input_box,
                multimask_output=False,
                hq_token_only=True,  # 关键：启用 HQ 模式
            )
            mask = (masks[0] > 0.5).astype(np.uint8) * 255
            results.append({
                'mask': mask,
                'bbox': [x1, y1, x2, y2],
                'area': int(np.sum(mask > 0))
            })
        return results

   
    def visualize(self, image_path, results, save_path=None, categories=None):
        import cv2
        import numpy as np
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定义颜色（每个实例不同颜色）
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (128, 0, 128),  # 紫色
            (255, 165, 0),  # 橙
            ]
    
        for i, res in enumerate(results):
            color = colors[i % len(colors)]
            mask = res['mask']      # 二值掩码 (0 或 255)
            bbox = res['bbox']
        
       
        # 创建一个和原图一样大的彩色图层
            colored_mask = np.zeros_like(image)
        # 把掩码区域设为对应的颜色
            colored_mask[mask > 0] = color
        
        # 半透明叠加 (alpha=0.5 表示透明度 50%)
            alpha = 0.5
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
     
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 2)
        
        #  保留：绘制边界框 
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 保留：添加类别标签 
            label = categories[i] if categories and i < len(categories) else f"#{i+1}"
        # 给标签加白色背景，更清晰
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (255, 255, 255), -1)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 保存结果
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"   保存: {save_path}")
    
        return image    
        pass