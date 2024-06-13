import cv2
import os
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager


class VideoProcessor:
    def __init__(self, ssim_threshold=0.8, images_per_second=2):
        self.ssim_threshold = ssim_threshold
        self.images_per_second = images_per_second

    def remove_exclude_xml(self):
        directory = './Dataset/jpg/labeled'

        # 獲取所有文件名
        files = os.listdir(directory)

        # 分離jpg和xml文件
        jpg_files = [f for f in files if f.endswith('.jpg')]
        xml_files = [f for f in files if f.endswith('.xml')]

        # 建立不含xml的jpg文件列表
        jpg_without_xml = [jpg for jpg in jpg_files if jpg.replace('.jpg', '.xml') not in xml_files]

        # 刪除不含xml的jpg文件
        for jpg in jpg_without_xml:
            os.remove(os.path.join(directory, jpg))
            print(f"Deleted {jpg}")

        print("不包含xml的jpg文件已刪除完成。")

    def calculate_ssim(self, imageA, imageB):
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        (score, _) = ssim(grayA, grayB, full=True)
        return score

    def process_single_video(self, args):
        file, queue = args
        if file.endswith('.MP4'):
            video_path = os.path.join('./Dataset/mp4', file)
            cap = cv2.VideoCapture(video_path)

            # 取得影片的每秒影格數（fps）
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"影片的每秒影格數（fps）: {fps}")

            # 依據fps計算每隔多少影格提取一次影像
            frame_interval = int(fps / self.images_per_second)

            # 取得影片的總影格數
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            image_count = 0

            # 儲存前一張影像，用於比較
            previous_frame = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if previous_frame is not None:
                        # 計算當前影像與前一張影像的 SSIM
                        ssim_score = self.calculate_ssim(previous_frame, frame)
                        if ssim_score < self.ssim_threshold:
                            # 若 SSIM 分數低於門檻，儲存當前影像
                            image_path = f'./Dataset/jpg/{file[:-4]}_{image_count}.jpg'
                            cv2.imwrite(image_path, frame)
                            image_count += 1
                    else:
                        # 若沒有前一張影像，直接儲存當前影像
                        image_path = f'./Dataset/jpg/{file[:-4]}_{image_count}.jpg'
                        cv2.imwrite(image_path, frame)
                        image_count += 1

                    # 更新前一張影像
                    previous_frame = frame

                    # 更新進度條
                    queue.put(1)

                frame_count += 1

            cap.release()

    def process_videos(self):
        # 獲取所有視頻文件
        video_files = [file for file in os.listdir('./Dataset/mp4') if file.endswith('.MP4')]
        manager = Manager()
        queue = manager.Queue()
        pool = Pool(cpu_count())

        total_frames = 0
        for file in video_files:
            video_path = os.path.join('./Dataset/mp4', file)
            cap = cv2.VideoCapture(video_path)
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // int(
                cap.get(cv2.CAP_PROP_FPS) / self.images_per_second
            )
            cap.release()

        # 使用多進程處理視頻文件
        args = [(file, queue) for file in video_files]
        pool_result = pool.map_async(self.process_single_video, args)

        with tqdm(total=total_frames, desc="Processing videos") as pbar:
            while not pool_result.ready():
                while not queue.empty():
                    pbar.update(queue.get())

        pool.close()
        pool.join()

    def copy_jpg_and_txt(self):
        source_directory = './Dataset/jpg'
        target_directory = './Dataset/jpg/yolo'

        # 確保目標目錄存在
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # 獲取所有jpg文件
        jpg_files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]

        for jpg in jpg_files:
            txt_file = jpg.replace('.jpg', '.txt')
            jpg_path = os.path.join(source_directory, jpg)
            txt_path = os.path.join(source_directory, txt_file)

            if os.path.exists(txt_path):
                # 複製jpg和txt文件到目標目錄
                shutil.copy(jpg_path, target_directory)
                shutil.copy(txt_path, target_directory)
                print(f"Copied {jpg} and {txt_file} to {target_directory}")

        print("包含txt的jpg及其txt已複製完成。")


if __name__ == '__main__':
    processor = VideoProcessor()
    # processor.process_videos()
    processor.copy_jpg_and_txt()
    # processor.remove_exclude_xml()
