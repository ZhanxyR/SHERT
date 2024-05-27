import cv2
import numpy as np

def write_pic(path, img, type=1):
    # -1 ~ 1
    if type == 1:
        img = img * 127.5 + 127.5
    # 0 ~ 1
    elif type == 2:
        img = img * 255 

    img = np.array(img)
    cv2.imwrite(path, img)

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return mask / 255

def inverse_mask(img):
    img = img.copy()
    img[img!=0] = 1
    img += 1
    img[img==2] = 0

    return img

