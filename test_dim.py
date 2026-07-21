# test_dim.py
from sentence_transformers import SentenceTransformer
import torch

# 加载模型
model_path = r"C:\Users\29941\Desktop\albb\FineGrained-Vision-Module\models\all-mpnet-base-v2"
model = SentenceTransformer(model_path)

# 测试文本
text = "这件衣服的领口"
embedding = model.encode(text)

print(f"文本: {text}")
print(f"特征维度: {embedding.shape[0]}")
print(f"特征前5维: {embedding[:5]}")