import os
import shutil
import random
from pathlib import Path

SOURCE = 'dataset_beras'
TARGET = 'dataset_beras_split'
SPLIT_RATIO = 0.8  

for varietas in os.listdir(SOURCE):
    src_path = os.path.join(SOURCE, varietas)
    if not os.path.isdir(src_path):
        continue

    images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for phase, image_list in [('train', train_images), ('val', val_images)]:
        dst_dir = os.path.join(TARGET, phase, varietas)
        os.makedirs(dst_dir, exist_ok=True)
        for img in image_list:
            shutil.copy(os.path.join(src_path, img), os.path.join(dst_dir, img))

print("✅ Dataset berhasil di-split ke folder dataset_beras_split/")
