# src/models/language_guided_localizer.py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os


class LanguageGuidedLocalizer:
    def __init__(self, text_encoder_type="qwen"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_encoder_type = text_encoder_type

        # 加载 DINOv2
        print(" 加载 DINOv2...")
        local_model_path = r"C:\Users\29941\Desktop\albb\model\dinov2"
        if os.path.exists(os.path.join(local_model_path, "model.safetensors")):
            print(f"   从本地加载: {local_model_path}")
            from transformers import AutoModel
            self.dinov2 = AutoModel.from_pretrained(local_model_path, trust_remote_code=True)
            self.dinov2.to(self.device)
            self.dinov2.eval()
            print(f" DINOv2 加载完成，设备: {self.device}")
        else:
            print(f" 找不到模型文件: {local_model_path}")
            raise FileNotFoundError(f"请确认路径是否正确: {local_model_path}")

        #加载文本编码器(384 维)
        self.text_encoder = None
        if text_encoder_type == "qwen":
            self._load_qwen()
        else:
            self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            print("加载 SentenceTransformer（768维）...")
            # 使用本地 768 维模型
            local_model_path = r"C:\Users\29941\Desktop\albb\FineGrained-Vision-Module\models\all-mpnet-base-v2"
            self.text_encoder = SentenceTransformer(local_model_path)
            print("SentenceTransformer 加载完成（768维）")
        except ImportError:
            print("请安装 sentence-transformers")
            raise

    def _load_qwen(self):
        try:
            from transformers import QwenVLForConditionalGeneration, QwenVLProcessor
            import sys
            sys.path.append('.')
            from configs.paths import QWEN_VL_MODEL
            print(" 加载 Qwen-VL...")
            self.qwen_processor = QwenVLProcessor.from_pretrained(QWEN_VL_MODEL)
            self.qwen_model = QwenVLForConditionalGeneration.from_pretrained(
                QWEN_VL_MODEL,
                device_map='auto',
                torch_dtype=torch.float16
            )
            print(" Qwen-VL 加载完成")
        except Exception as e:
            print(f" Qwen-VL 加载失败: {e}")
            print("   将使用 sentence-transformers 作为备用")
            self._load_sentence_transformer()

    def extract_visual_features(self, image_path, regions):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        features = []
        for bbox in regions:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1+1, x2), max(y1+1, y2)

            crop = image.crop((x1, y1, x2, y2))
            input_tensor = transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.dinov2(input_tensor)
                if hasattr(outputs, 'last_hidden_state'):
                    feat = outputs.last_hidden_state[:, 0, :]
                else:
                    feat = outputs
            features.append(feat.squeeze().cpu().numpy())

        return np.array(features)

    def encode_text(self, text):
        if self.text_encoder is None:
            raise RuntimeError("文本编码器未加载")
        return self.text_encoder.encode(text)

    def localize(self, image_path, query, regions):
        if not regions:
            return None

        text_feat = self.encode_text(query)
        if text_feat is None:
            return None

        visual_feats = self.extract_visual_features(image_path, regions)
        if len(visual_feats) == 0:
            return None

        similarities = []
        for vf in visual_feats:
            sim = np.dot(text_feat, vf) / (np.linalg.norm(text_feat) * np.linalg.norm(vf) + 1e-8)
            similarities.append(float(sim))

    # 检查 similarities 是否为空
        if not similarities:
            return None

        best_idx = int(np.argmax(similarities))

        return {
            'best_region_idx': best_idx,
            'best_bbox': regions[best_idx],
            'similarities': similarities,
            'confidence': similarities[best_idx]
    }
        