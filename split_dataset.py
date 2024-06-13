import os
import shutil
import random


def split_dataset(source_dir, images_dir, labels_dir, train_ratio=0.7, val_ratio=0.2):
    # 確保目標資料夾存在
    os.makedirs(images_dir + '/train', exist_ok=True)
    os.makedirs(images_dir + '/val', exist_ok=True)
    os.makedirs(images_dir + '/test', exist_ok=True)
    os.makedirs(labels_dir + '/train', exist_ok=True)
    os.makedirs(labels_dir + '/val', exist_ok=True)
    os.makedirs(labels_dir + '/test', exist_ok=True)

    # 獲取所有jpg文件
    jpg_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    # 檢查每個jpg文件是否有對應的txt文件，沒有則刪除jpg文件
    for jpg in jpg_files:
        txt = jpg.replace('.jpg', '.txt')
        if not os.path.exists(os.path.join(source_dir, txt)):
            os.remove(os.path.join(source_dir, jpg))
            print(f"已刪除沒有對應txt文件的jpg文件: {jpg}")

    # 重新獲取所有jpg文件
    jpg_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

    # 按比例分割資料
    random.shuffle(jpg_files)
    train_index = int(len(jpg_files) * train_ratio)
    val_index = int(len(jpg_files) * (train_ratio + val_ratio))
    train_files = jpg_files[:train_index]
    val_files = jpg_files[train_index:val_index]
    test_files = jpg_files[val_index:]

    # 移動文件到相應資料夾
    for jpg in train_files:
        txt = jpg.replace('.jpg', '.txt')
        shutil.move(os.path.join(source_dir, jpg), os.path.join(images_dir, 'train', jpg))
        shutil.move(os.path.join(source_dir, txt), os.path.join(labels_dir, 'train', txt))

    for jpg in val_files:
        txt = jpg.replace('.jpg', '.txt')
        shutil.move(os.path.join(source_dir, jpg), os.path.join(images_dir, 'val', jpg))
        shutil.move(os.path.join(source_dir, txt), os.path.join(labels_dir, 'val', txt))

    for jpg in test_files:
        txt = jpg.replace('.jpg', '.txt')
        shutil.move(os.path.join(source_dir, jpg), os.path.join(images_dir, 'test', jpg))
        shutil.move(os.path.join(source_dir, txt), os.path.join(labels_dir, 'test', txt))

    print("資料集已分割完成。")


# 定義路徑
source_directory = './Dataset/jpg/yolo'
images_directory = './Dataset/jpg/yolo/images'
labels_directory = './Dataset/jpg/yolo/labels'

# 分割資料集
split_dataset(source_directory, images_directory, labels_directory)
