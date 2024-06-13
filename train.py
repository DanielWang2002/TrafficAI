import os
from ultralytics import YOLO


# 訓練函式
def train_yolov8(data_yaml, model_name='yolov8n.pt', epochs=100, lr=0.001, imgsz=1280):
    model = YOLO(model_name)  # 載入預訓練模型
    model.train(
        data=data_yaml,
        epochs=epochs,
        device='mps',
        lr0=lr,
        imgsz=imgsz,
        box=0.05,  # 調整Box損失權重
        cls=0.5,  # 調整分類損失權重
        dfl=1.5,  # 調整DFL損失權重
        verbose=True,
    )  # 開始訓練

    # 儲存訓練後的模型
    model.save('trained_model.pt')


# 測試函式
def test_yolov8(data_yaml, model_path):
    model = YOLO(model_path)  # 載入訓練後的模型
    results = model.val(data=data_yaml, split='test')  # 使用測試集進行評估
    print(results)


if __name__ == '__main__':
    data_yaml_path = './trafficai.yaml'
    train_yolov8(data_yaml_path, lr=0.001, imgsz=1280)  # 調整學習率和輸入圖片分辨率

    # 訓練完成後進行測試
    trained_model_path = 'trained_model.pt'
    test_yolov8(data_yaml_path, trained_model_path)
