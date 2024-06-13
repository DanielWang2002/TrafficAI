import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


def convert_voc_to_yolo(voc_path, yolo_path):
    tree = ET.parse(voc_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_annotations = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        class_id = name_to_class_id(name)  # 自行實現一個名稱到類別ID的對應函數

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {w} {h}")

    with open(yolo_path, 'w') as f:
        f.write("\n".join(yolo_annotations))


def name_to_class_id(name):
    class_mapping = {
        'l0': 0,
        'l1': 1,
        'r0': 2,
        'r1': 3,
    }
    return class_mapping[name]


# 設定目錄路徑
input_directory = './Dataset/jpg/labeled'
output_directory = './Dataset/jpg/yolo'

# 確保輸出目錄存在
os.makedirs(output_directory, exist_ok=True)

# 讀取目錄下的所有XML文件
xml_files = [f for f in os.listdir(input_directory) if f.endswith('.xml')]
print(len(xml_files))

# 進行轉換並顯示進度條
for xml_file in tqdm(xml_files, desc="Converting XML to YOLO format"):
    voc_path = os.path.join(input_directory, xml_file)
    yolo_file = xml_file.replace('.xml', '.txt')
    yolo_path = os.path.join(output_directory, yolo_file)

    convert_voc_to_yolo(voc_path, yolo_path)
