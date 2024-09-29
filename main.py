####################################### import模块 #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys

from fastapi import FastAPI, File, status
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from io import BytesIO

from app import get_image_from_bytes
from app import detect_sample_model
from app import add_bboxs_on_img
from app import get_bytes_from_image

####################################### 日志 #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI 设置 #############################

# 标题
app = FastAPI(
    title="Object Detection FastAPI 模板",
    description="""从图像中获取对象值
                    并返回图像和JSON结果""",
    version="2023.1.31",
)

# 如果您希望允许来自特定域（在origins参数中指定）的客户端请求
# 访问FastAPI服务器的资源，并且客户端和服务器托管在不同的域上，则需要此功能。
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def save_openapi_json():
    '''此函数用于将FastAPI应用程序的OpenAPI文档数据保存到JSON文件中。
    保存OpenAPI文档数据的目的是拥有API规范的永久和离线记录，
    可用于文档目的或生成客户端库。虽然不一定需要，但在某些情况下可能会有帮助。'''
    openapi_data = app.openapi()
    # 将"openapi.json"更改为所需的文件名
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)


# 重定向
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    它发送一个GET请求到该路由，并希望得到一个"200"响应代码。
    未能返回200响应代码将使GitHub Actions回滚到项目处于"工作状态"的最后一个版本。
    它作为最后一道防线，以防发生问题。
    此外，它还以JSON格式返回响应，形式为：
    {
        'healthcheck': '一切正常！'
    }
    '''
    return {'healthcheck': '一切正常！'}


######################### 支持函数 #################################

def crop_image_by_predict(image: Image, predict: pd.DataFrame(), crop_class_name: str, ) -> Image:
    """根据图像中某个对象的检测结果裁剪图像。

    参数:
        image: 要裁剪的图像。
        predict (pd.DataFrame): 包含对象检测模型预测结果的数据框。
        crop_class_name (str, 可选): 要根据其裁剪图像的对象类名称。如果未提供，函数将返回图像中找到的第一个对象。

    返回:
        Image: 裁剪后的图像或None
    """
    crop_predicts = predict[(predict['name'] == crop_class_name)]

    if crop_predicts.empty:
        raise HTTPException(status_code=400, detail=f"照片中未找到{crop_class_name}")

    # 如果有多个检测结果，选择置信度更高的那个
    if len(crop_predicts) > 1:
        crop_predicts = crop_predicts.sort_values(by=['confidence'], ascending=False)

    crop_bbox = crop_predicts[['xmin', 'ymin', 'xmax', 'ymax']].iloc[0].values
    # 裁剪
    img_crop = image.crop(crop_bbox)
    return (img_crop)


######################### 主功能 #################################

@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    从图像中进行对象检测。

    参数:
        file (bytes): 以字节格式的图像文件。
    返回:
        dict: 包含对象检测结果的JSON格式。
    """
    # 步骤1：用None值初始化结果字典
    result = {'detect_objects': None}

    # 步骤2：将图像文件转换为图像对象
    input_image = get_image_from_bytes(file)

    # 步骤3：从模型中进行预测
    predict = detect_sample_model(input_image)

    # 步骤4：选择检测对象返回信息
    # 您可以在此选择要发送到结果中的数据
    detect_res = predict[['name', 'confidence']]
    objects = detect_res['name'].values

    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))

    # 步骤5：日志记录和返回
    logger.info("结果: {}", result)
    return result


@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    从图像中进行对象检测并在图像上绘制边界框

    参数:
        file (bytes): 以字节格式的图像文件。
    返回:
        Image: 带有边界框注释的字节格式图像。
    """
    # 从字节获取图像
    input_image = get_image_from_bytes(file)

    # 模型预测
    predict = detect_sample_model(input_image)

    # 在图像上添加边界框
    final_image = add_bboxs_on_img(image=input_image, predict=predict)

    # 以字节格式返回图像
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")




#   uvicorn main:app --reload --host 0.0.0.0 --port 8001