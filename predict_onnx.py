import cv2 as cv
import numpy as np
from openvino.runtime import Core


# load model
# labels = load_classes()
abels = ["person"]
ie = Core()
for device in ie.available_devices:
    print(device)
model = ie.read_model(model="yolov10n.onnx")
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)

frame = cv.imread("yzr.jpg")
image = format_yolov10(frame)

h, w, c = image.shape
x_factor = w / 640.0
y_factor = h / 640.0

# 检测 2/255.0, NCHW = 1x3x640x640
blob = cv.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)

# 设置网络输入
cvOut = compiled_model([blob])[output_layer]
# [left,top, right, bottom, score, classId]
print(cvOut.shape)
for row in cvOut[0,:,:]:
    score = float(row[4])
    objIndex = int(row[5])
    if score > 0.5:
        left, top, right, bottom = row[0].item(), row[1].item(), row[2].item(), row[3].item()

        left = int(left * x_factor)
        top = int(top * y_factor)
        right = int(right * x_factor)
        bottom = int(bottom * y_factor)
        # 绘制
        cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
        cv.putText(frame, "score:%.2f, %s"%(score, labels[objIndex]),
                (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8);

cv.imshow('YOLOv10 Object Detection', frame)
cv.imwrite("D:/result.png", frame)
cv.waitKey(0)
cv.destroyAllWindows()
