import os
import random
import cv2


def draw_label(img, label_path, class_names):
    # 讀取標注檔案
    with open(label_path, 'r') as file:
        labels = file.readlines()

    height, width, _ = img.shape

    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # 繪製矩形框和類別標籤
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )


def display_random_image_with_label(images_dir, labels_dir, class_names):
    # 獲取所有圖片檔案
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    # 隨機抽選一個圖片檔案
    random_image = random.choice(images)
    image_path = os.path.join(images_dir, random_image)
    label_path = os.path.join(labels_dir, random_image.replace('.jpg', '.txt'))

    # 讀取圖片
    img = cv2.imread(image_path)

    # 繪製標注
    draw_label(img, label_path, class_names)

    # 顯示圖片
    cv2.imshow("Random Image with Labels", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    images_dir = './Dataset/jpg/yolo/images/train'
    labels_dir = './Dataset/jpg/yolo/labels/train'
    class_names = ['l0', 'l1', 'r0', 'r1']

    display_random_image_with_label(images_dir, labels_dir, class_names)
