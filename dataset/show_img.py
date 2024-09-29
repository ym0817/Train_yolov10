import glob
import os
import cv2







def show_img(path):
    img_dir = os.path.join(path,'images')
    lab_dir = os.path.join(path, 'xmls_txt')

    for img_path in glob.glob(img_dir + '/*'):
        file_name = os.path.basename(img_path).split('.')[0]
        im = cv2.imread(img_path)
        img_height,img_width,c = im.shape
        label_file = os.path.join(lab_dir, file_name + '.txt')
        label_lines = open(label_file,'r').readlines()
        print(img_path, len(label_lines))
        for label_line in label_lines:
            words = label_line.strip().split(' ')
            parts = [eval(i) for i in words]
            c= parts[0]


            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            # 计算边界框坐标
            x_min = int((x - w / 2) * img_width)
            y_min = int((y - h / 2) * img_height)
            x_max = int((x + w / 2) * img_width)
            y_max = int((y + h / 2) * img_height)




            print((x_min, y_min), (x_max, y_max))
            cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        cv2.imshow('re', im)
        cv2.waitKey(0)



if __name__ == '__main__':

    test_path = 'Val'
    show_img(test_path)










