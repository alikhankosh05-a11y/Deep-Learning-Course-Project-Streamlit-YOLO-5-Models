import os
import shutil
import random
from collections import defaultdict

# ================= НАСТРОЙКИ =================
data_root = '/Users/allikhankoshamet/Desktop/dl_project/dl_new_dataset'  # ← твоя папка
train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15
seed = 42
# =============================================

random.seed(seed)

# Создаём папки
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(data_root, split), exist_ok=True)

# Находим все классы (папки с картинками)
classes = [d for d in os.listdir(data_root) 
           if os.path.isdir(os.path.join(data_root, d)) 
           and d not in ['train', 'val', 'test']]

print(f"Найдено классов: {len(classes)} → {classes}\n")

for cls in classes:
    cls_path = os.path.join(data_root, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    total = len(images)
    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    
    train_imgs = images[:train_size]
    val_imgs   = images[train_size:train_size+val_size]
    test_imgs  = images[train_size+val_size:]
    
    # Создаём папки классов внутри train/val/test
    for split, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        split_cls_path = os.path.join(data_root, split, cls)
        os.makedirs(split_cls_path, exist_ok=True)
        
        for img in img_list:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_cls_path, img)
            shutil.copy(src, dst)
    
    print(f"{cls:20} → train:{len(train_imgs):4}  val:{len(val_imgs):3}  test:{len(test_imgs):3}")

print("\n✅ Разбивка завершена!")
print(f"Структура готова: {data_root}/train , /val , /test")