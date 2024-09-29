from PIL import Image
import io
import pandas as pd
import numpy as np

from typing import Optional

from ultralytics import YOLOv10
from ultralytics.utils.plotting import Annotator, colors

# 初始化模型
model_sample_model = YOLOv10("./models/sample_model/yolov10n.pt")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """将字节格式的图像转换为PIL RGB格式

    参数:
        binary_image (bytes): 图像的字节表示

    返回:
        PIL.Image: PIL RGB格式的图像
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    将PIL图像转换为字节

    参数:
    image (Image): PIL图像实例

    返回:
    bytes : 包含JPEG格式图像的BytesIO对象，质量为85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # 将图像保存为JPEG格式，质量为85
    return_image.seek(0)  # 将指针设置到文件开头
    return return_image


def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    将yolov8的预测结果（torch.Tensor）转换为pandas DataFrame。

    参数:
        results (list): 包含yolov8预测输出的列表，形式为torch.Tensor。
        labeles_dict (dict): 包含标签名称的字典，其中键为类id，值为标签名称。

    返回:
        predict_bbox (pd.DataFrame): 包含边界框坐标、置信度分数和类别标签的DataFrame。
    """
    # 将Tensor转换为numpy数组
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    # 将预测的置信度添加到DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # 将预测的类别添加到DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # 用labeles_dict中的类名替换类编号
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox


def get_model_predict(model: YOLOv10, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5,
                      augment: bool = False) -> pd.DataFrame:
    """
    获取模型对输入图像的预测结果。

    参数:
        model (YOLO): 训练好的YOLO模型。
        input_image (Image): 模型将进行预测的图像。
        save (bool, 可选): 是否保存带有预测结果的图像。默认为False。
        image_size (int, 可选): 模型接收的图像大小。默认为1248。
        conf (float, 可选): 预测的置信度阈值。默认为0.5。
        augment (bool, 可选): 是否对输入图像应用数据增强。默认为False。

    返回:
        pd.DataFrame: 包含预测结果的DataFrame。
    """
    # 进行预测
    predictions = model.predict(
        imgsz=image_size,
        source=input_image,
        conf=conf,
        save=save,
        augment=augment,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
    )

    # 将预测结果转换为pandas DataFrame
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


################################# 边界框功能 #####################################

def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    在图像上添加边界框

    参数:
    image (Image): 输入图像
    predict (pd.DataFrame): 模型的预测结果

    返回:
    Image: 带有边界框的图像
    """
    # 创建注释器对象
    annotator = Annotator(np.array(image))

    # 按xmin值对预测结果进行排序
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # 迭代预测结果的每一行
    for i, row in predict.iterrows():
        # 创建要显示在图像上的文本
        text = f"{row['name']}: {int(row['confidence'] * 100)}%"
        # 获取边界框坐标
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # 在图像上添加边界框和文本
        annotator.box_label(bbox, text, color=colors(row['class'], True))
    # 将带注释的图像转换为PIL图像
    return Image.fromarray(annotator.result())


################################# 模型 #####################################


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    使用sample_model进行预测。
    基于YoloV10

    参数:
        input_image (Image): 输入图像。

    返回:
        pd.DataFrame: 包含对象位置的DataFrame。
    """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict