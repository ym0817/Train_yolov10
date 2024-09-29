from email.contentmanager import maintype

import cv2
import os
import shutil
from gradio.themes.builder_app import main_fonts


def rm_useless_imgs(path, mode):
    dirs = os.listdir(path)
    cnt = 0
    img_path = os.path.join(path, "images", mode)
    label_path = os.path.join(path, "labels", mode)
    imgs = os.listdir(img_path)
    for img_name in imgs:
        imgfile_path = os.path.join(img_path, img_name)
        cvim = cv2.imread(imgfile_path)
        if cvim is None:
            cnt += 1
            labelfile_path = os.path.join(label_path, img_name).replace('.jpg', '.txt')
            print(imgfile_path)
            os.remove(imgfile_path)
            os.remove(labelfile_path)



    print("cnt", cnt)








if __name__ == '__main__':
    img_dir = "NEU-DET"
    # type = "train"
    type = "test"
    rm_useless_imgs(img_dir, type)

