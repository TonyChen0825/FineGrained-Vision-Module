# scripts/extract_samples.py
import sys
import os
import json
import shutil
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.paths import DEEPFASHION2_ROOT, SAMPLES_DIR, SAMPLES_IMAGES_DIR


def extract_samples():
    """从DeepFashion2提取100张图片到项目内"""
    
    annos_dir = os.path.join(DEEPFASHION2_ROOT, "train", "annos")
    images_dir = os.path.join(DEEPFASHION2_ROOT, "train", "image")
    
    if not os.path.exists(annos_dir):
        print(f" 找不到标注文件夹: {annos_dir}")
        return
    
    if not os.path.exists(images_dir):
        print(f" 找不到图片文件夹: {images_dir}")
        return
    
    json_files = glob.glob(os.path.join(annos_dir, "*.json"))
    json_files.sort()
    
    print(f" 找到 {len(json_files)} 个标注文件")
    
    # 8类服饰
    cat_names = {
        1: '上衣', 2: '长袖上衣', 3: '短袖外套', 4: '长袖外套',
        5: '背心', 6: '吊带', 7: '短裤', 8: '长裤', 9: '裙子', 
        10: '短袖连衣裙', 11: '长袖连衣裙', 12: '背心连衣裙', 13: '吊带连衣裙'
    }
    
    selected = []
    img_count = 0
    
    print(" 提取图片...")
    
    for json_path in json_files:
        if img_count >= 1000:
            break
        
        with open(json_path, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
        
        # 提取所有 item 中的标注信息
        annotations = []
        for key, value in ann_data.items():
           
            if key.startswith('item'):
                cat_id = value.get('category_id', 0)
                cat_name = value.get('category_name', '未知')
                bbox = value.get('bounding_box', [])
                
                # 只保留8类服饰
                if cat_id in cat_names:
                   
                    # 从数据看: [249, 423, 466, 623]  [x1, y1, x2, y2]
                    if len(bbox) == 4:
                        annotations.append({
                            'bbox': bbox,
                            'category_id': cat_id,
                            'category_name': cat_names.get(cat_id, cat_name),
                            'segmentation': value.get('segmentation', [])
                        })
        
        if not annotations:
            continue
        
        # 复制图片
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        src_img = os.path.join(images_dir, f"{base_name}.jpg")
        dst_img = os.path.join(SAMPLES_IMAGES_DIR, f"{base_name}.jpg")
        
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            selected.append({
                "image_id": base_name,
                "file_name": f"{base_name}.jpg",
                "annotations": annotations
            })
            img_count += 1
            
            if img_count % 10 == 0:
                print(f"  已提取 {img_count} 张...")
    
    # 保存标注
    ann_save_path = os.path.join(SAMPLES_DIR, "annotations.json")
    with open(ann_save_path, 'w', encoding='utf-8') as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    
    total_ann = sum(len(s['annotations']) for s in selected)
    print(f"\n 提取完成！")
    print(f"  图片数: {len(selected)}")
    print(f"  标注数: {total_ann}")
    print(f"  保存位置: {SAMPLES_DIR}")
    
    cat_count = {}
    for s in selected:
        for ann in s['annotations']:
            name = ann['category_name']
            cat_count[name] = cat_count.get(name, 0) + 1
    print("\n 各类别分布:")
    for name, count in sorted(cat_count.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    extract_samples()