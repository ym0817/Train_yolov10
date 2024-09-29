import cv2
import requests
import numpy as np
from flask import Flask, request, jsonify, Blueprint
import urllib
from ultralytics import YOLOv10

app = Flask(__name__)
import urllib.parse

# 创建蓝图
blueprint = Blueprint('my_blueprint', __name__, url_prefix='102.168.67.16')  # 地址名自己起，例如/detect


def read_img_from_url(url):
    try:
        # 下载图片
        response = urllib.request.urlopen(url)
        image_data = response.read()
        # 将图像数据解码成opencv可以读取的格式
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print("Error occurred while reading image from URL:", str(e))
        return None


@blueprint.route('地址2', methods=['GET', 'POST'])  # 同地址1的取名
def detect():
    if request.method == 'POST':
        # 获取前端发送的JSON数据
        data = request.json
        image_url = data['url']
        source_image = read_img_from_url(image_url)
        model = YOLOv10("ultralytics/cfg/models/v10/yolov10n.yaml")
        model = YOLOv10("yolov10n.pt")
        results = model.predict(source_image)
        response = {
            'results': results
        }
        return jsonify(response)
    elif request.method == 'GET':
        return "Please send POST requests!"


# 在应用程序上注册蓝图
app.register_blueprint(blueprint)
if __name__ == '__main__':
    # 运行Flask应用
    app.run(host="0.0.0.0", port=8000, debug=True)