import numpy as np
import cv2

from utils import rescaleImage, generateMask

def preprocess(path : str, id : str) -> None : 
    background_path = path + id + "_background.jpg"
    original1_path = path + id + "_original copy.jpg"
    original_path = path + id + "_original.jpg"
    rescaled_path = path + id + "_rescaled.jpg"
    mask_path = path + id + "_mask.jpg"
    source_path = path + id + "_source.jpg"

    #background_image = cv2.imread(background_path) 
    #print(background_image.shape)
#    print(rescaleImage(background_path, rescaled_path, (100, 200), 1)) #a/b
#    print(rescaleImage(background_path, rescaled_path, (100, 800), 1)) #c
#    print(rescaleImage(background_path, rescaled_path, (100, 800), 1)) #d
#    print(rescaleImage(background_path, rescaled_path, (100, 400), 1)) #e
#    print(rescaleImage(background_path, rescaled_path, (100, 500), 1)) #f
    print(rescaleImage(background_path, rescaled_path, (100, 250), 1)) #g
#    print(rescaleImage(background_path, rescaled_path, (100, 250), 1)) #h
    #rescaled_image = cv2.imread(rescaled_path)

    original_image = cv2.imread(original_path)
#    print(rescaleImage(original1_path, original_path, (100, 200), 1)) #a/bs

#    print(rescaleImage(original1_path, original_path, (100, 80), 1)) #c
#    print(rescaleImage(original1_path, original_path, (100, 400), 1)) #d
#    print(rescaleImage(original1_path, original_path, (100, 200), 1)) #e
#    print(rescaleImage(original1_path, original_path, (100, 300), 1)) #f
    print(rescaleImage(original1_path, original_path, (100, 100), 1)) #g
#    print(rescaleImage(original1_path, original_path, (100, 250), 1)) #h

    #print(rescaled_image.shape)
    #print(original_image.shape)
    generateMask(original_path)

    original_image = cv2.imread(original_path)
    mask = cv2.imread(mask_path)
    source_image = cv2.bitwise_and(original_image, mask)
    cv2.imwrite(source_path, source_image)

def removeEdge(path, id) -> None :
    mask1_path = path + id + "_mask.jpg"
    mask2_path = path + id + "_mask_edited.jpg"
    source1_path = path + id + "_source.jpg"
    source2_path = path + id + "_source_edited.jpg"
    ori1_path = path + id + "_original.jpg"
    ori2_path = path + id + "_original_edited.jpg"
    mask1 = cv2.imread(mask1_path)
    source1 = cv2.imread(source1_path)
    ori1 = cv2.imread(ori1_path)
    h, w = mask1.shape[0:2]
    lx, ly = h, w
    rx, ry = 0, 0
    for y in range(h):
        for x in range(w):
            if mask1[y][x][0] == 255 and mask1[y][x][1] == 255 and mask1[y][x][2] == 255:
                ly = min(ly, y)
                lx = min(lx, x)
                ry = max(ry, y)
                rx = max(rx, x)
    print(ly, lx, ry, rx)
    mask2 = np.zeros((ry-ly+1, rx-lx+1), dtype = np.uint8)
    mask2 = mask1[ly : ry+1, lx : rx+1]
    source2 = source1[ly : ry+1, lx : rx+1]
    ori2 = ori1[ly-1 : ry+2, lx-1 : rx+2]
    cv2.imwrite(mask2_path, mask2)
    cv2.imwrite(source2_path, source2)
    cv2.imwrite(ori2_path, ori2)
    