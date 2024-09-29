# import warnings
#
# warnings.filterwarnings('ignore')
from ultralytics import YOLOv10

# data_yaml = "/home/ymm/Works/yolov10/coco128.yaml"
data_yaml = './NEU-DET.yaml'

if __name__ == '__main__':
    model = YOLOv10('ultralytics/cfg/models/v10/yolov10n.yaml')
    model.load('yolov10n.pt')  # loading pretrain weights
    model.train(data= data_yaml,
                imgsz=640,
                epochs=100,
                batch=8,
                close_mosaic=0,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='exp',
                resume=True,  # 断点训练，默认Flase
                plots = True
        )