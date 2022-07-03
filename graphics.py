import cv2
import numpy as np
from sqlalchemy import true

direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]
# A function to return the list of all the neighbors of the pixel
def getNeighborhood(pixel : tuple, dimension : tuple) -> list:
    pixel_y, pixel_x = pixel
    h, w = dimension
    neighborhood = []
    for d in range(4):
        neigh_y = pixel_y + direction[d][0]
        neigh_x = pixel_x + direction[d][1]
        if neigh_y >= 0 and neigh_y < h and neigh_x >= 0 and neigh_x < w:
            neighborhood.append((neigh_y, neigh_x))
    return neighborhood

def getBoundary(mask : np.ndarray,
                maskPixels : list
    ) -> tuple:
    dimension = mask.shape
    boundary = np.zeros(dimension, dtype = np.uint8)
    boundaryPixels = []
    for pixel in maskPixels:
        neighborhood = getNeighborhood(pixel, dimension)
        pixel_y, pixel_x = pixel
        flag = False
        for neighbor in neighborhood:
            if not mask[neighbor]:
                flag = True
                break
        if flag :
            boundary[pixel] = 1
            boundaryPixels.append(pixel)
    return boundary, boundaryPixels

# A function to give every pixel a label in some order and 
# return the list of the pixels in the mask, and the mapping from mask pixel to index
def labelMask(mask : np.ndarray     # the matrix of bits describing the mask
    ) -> tuple:     # tuple of (list, dict)
    maskPixels = []
    pixel2idx = dict()
    count = 0
    h, w = mask.shape
    for y in range(h) :
        for x in range(w) :
            if mask[y][x] :
                maskPixels.append((y, x))
                pixel2idx[(y, x)] = count
                count += 1
    return maskPixels, pixel2idx
