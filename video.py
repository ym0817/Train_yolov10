import cv2
from ultralytics import YOLOv10

model = YOLOv10("yolov10n.pt")
cap = cv2.VideoCapture('test_t.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有读取到帧，退出循环
    results = model.predict(frame)
    # 遍历每个预测结果
    for result in results:
        # 结果中的每个元素对应一张图片的预测
        boxes = result.boxes  # 获取边界框信息
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)
    # 显示带有检测结果的帧
    cv2.imshow('YOLOv10', frame)
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
