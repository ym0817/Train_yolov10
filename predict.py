import cv2
from ultralytics import YOLOv10 # Note the "v10" in the end


imgpaths = ['1.jpg', '2.jpg', '3.jpg']

# Load a model
model = YOLOv10('runs/train/exp/weights/best.pt') # load an official model
# Predict with the model
# model.predict(0) # predict on your webcam

results = model.predict(imgpaths[0]) # predict on your webcam
results[0].show()


# results = model(source=imgpaths, conf=0.25,save=True)


import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TkAgg')

images = glob.glob('runs/detect/predict/*.jpg')

images_to_display = images[:2]

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for i, ax in enumerate(axes.flat):
    if i < len(images_to_display):
        img = mpimg.imread(images_to_display[i])
        ax.imshow(img)
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
