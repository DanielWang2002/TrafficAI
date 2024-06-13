import os
import shutil


def move_files_back(src_dirs, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for src_dir in src_dirs:
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.txt'):
                    shutil.move(os.path.join(root, file), os.path.join(dst_dir, file))


# 定義原來的目錄
# src_dirs = [
#     './Dataset/jpg/yolo/train/images',
#     './Dataset/jpg/yolo/train/labels',
#     './Dataset/jpg/yolo/val/images',
#     './Dataset/jpg/yolo/val/labels',
#     './Dataset/jpg/yolo/test/images',
#     './Dataset/jpg/yolo/test/labels',
# ]
src_dirs = [
    './Dataset/jpg/yolo/images/train',
    './Dataset/jpg/yolo/images/test',
    './Dataset/jpg/yolo/images/val',
]

# 定義目標目錄
dst_dir = './Dataset/jpg/yolo'

# 移動文件回原來的目錄
move_files_back(src_dirs, dst_dir)
