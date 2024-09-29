from ultralytics import YOLOv10


def main(modelpath, datapath):
    # 加载模型，split='test'利用测试集进行测试
    model = YOLOv10(modelpath)
    model.val(data=datapath,
              # split='test',
              imgsz=640,
              batch=8,
              device=0,
              workers=8)  # 模型验证


if __name__ == "__main__":
    weight_path = 'runs/train/exp/weights/best.pt'
    data_path = 'NEU-DET.yaml'
    main(weight_path,data_path)
