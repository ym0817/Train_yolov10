from flask import Flask,request, jsonify
import cv2
from ultralytics import YOLOv10

app = Flask(__name__)
# det = YOLOv5()
model = YOLOv10('/home/ymm/Works/yolov10/runs/train/exp/weights/last.pt')


@app.route("/predict", methods=["GET","POST"])
def predict():
    result = {"success": False}
    if request.method == "POST":
        if request.files.get("image") is not None:
            try:
                # 得到客户端传输的图像
                start = time.time()
                input_image = request.files["image"].read()
                imBytes = np.frombuffer(input_image, np.uint8)
                iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
                # 执行推理
                results = model(iImage)
                print("duration: ", time.time() - start)

                if (outs is None) and (len(outs) < 0):
                    result["success"] = False
                for result in results:
                    # 结果中的每个元素对应一张图片的预测
                    boxes = result.boxes  # 获取边界框信息
                    # for box in boxes:
                    #     x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                    #     cls = int(box.cls[0])
                    #     conf = float(box.conf[0])
                    result["box"] = map(int, boxes[0].xyxy[0])
                    result["conf"] = float(boxes[0].conf[0])
                    result["classid"] = int(boxes[0].cls[0])
                    result['success'] = True


                # boxes = results[0].boxes
                # 将结果保存为json格式
                # result["box"] = outs[0].tolist()
                # result["conf"] = outs[1].tolist()
                # result["classid"] = outs[2].tolist()
                # result['success'] = True

            except Exception:
                pass

    return jsonify(result)


if __name__ == "__main__":
    print(("* Loading yolov5 model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(host='127.0.0.1', port=7000)