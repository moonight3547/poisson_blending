import cv2
from cv2 import Laplacian
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from graphics import getNeighborhood, labelMask, getBoundary
from linearalgebra import guideVec

'''
# Function to transfer the color(gray scale) ranging in [0, 255] into a bit [0/1] 
# i.e. 0 -> 0, 255 -> 1
def color2bit(color : int) -> int:
    return 1 if color >= 128 else 0
'''

# Solve the Laplacian equation at each mask pixel in the channel
def laplacianSolver(bkg : np.ndarray,       # the background image
                    src : np.ndarray,       # the source image
                    mask : np.ndarray,      # the mask (in bit [0/1])
                    maskPixels : list,      # the list of pixels in the mask
                    pixel2idx : dict,       # the mapping from mask pixel to index
                    boundary : np.ndarray,  # the boundary of the mask
                    boundaryPixels : list,  # the list of pixels on the boundary
                    offset : tuple          # offset between source image and corresponding position in the background
    ) -> np.ndarray :
    
    blended_image = bkg.copy()
    n = len(maskPixels)
    A = lil_matrix(np.zeros((n, n)))
    b = np.zeros((n, 1))
    dimension = mask.shape

    for i, pixel in enumerate(maskPixels):
        neighborhood = getNeighborhood(pixel, dimension)
        A[i, i] = len(neighborhood)
        rhs = 0
        for neighbor in neighborhood:
            if mask[neighbor] == 1:
                A[i, pixel2idx[neighbor]] = -1
            else :
                rhs += bkg[neighbor]
            rhs += guideVec(pixel, neighbor, bkg, src, offset)
        b[i, 0] = rhs
    print('Matrix Constructed.')
    x = spsolve(csc_matrix(A), b)
    x = np.maximum(x ,0)
    x = np.minimum(x, 255)
    for i, pixel in enumerate(maskPixels):
        blended_image[pixel] = np.uint8(x[i])
    print('Linear Equation Solved!')
    return blended_image

# The main process of the Poisson Image Blending algorithm
# Blending by putting the masked part of source upon the corresponding position in the background 
def poissonImageBlending(path : str, id : str, offset : tuple):
    rescaled_path = path + id + "_rescaled.jpg"
    mask_path = path + id + "_mask_edited.jpg"
    mask2_path = path + id + "_mask_bkg.jpg"
    source_path = path + id + "_original_edited.jpg"
    result_path = path + id + "_blended.jpg"
    background = cv2.imread(rescaled_path)
    src_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    source = cv2.imread(source_path)
    assert(len(src_mask.shape) == 2)
    boundary_path = path + id + "_boundary.jpg"

    offset_h, offset_w = offset
    mask_h, mask_w = src_mask.shape
    src_mask = np.where(src_mask >= 128, 1, 0)
    mask = np.zeros(background.shape[0:2]) #, dtype = np.uint8)
    try :
        mask[offset_h : offset_h + mask_h, offset_w : offset_w + mask_w] = src_mask
    except Exception as e:
        print(e)
        print("You cannot put the source image into the background!")
    
    maskPixels, pixel2idx = labelMask(mask)
    boundary, boundaryPixels = getBoundary(mask, maskPixels)
    mask2 = mask
    mask2 = np.where(mask == 1, 255, 0)
    cv2.imwrite(mask2_path, mask2)
    cv2.imwrite(boundary_path, boundary)
#    print(len(boundaryPixels))
    
    blue_channel = laplacianSolver(background[:, :, 0], source[:, :, 0], 
                                    mask, maskPixels, pixel2idx, boundary, boundaryPixels, offset)
    green_channel = laplacianSolver(background[:, :, 1], source[:, :, 1], 
                                    mask, maskPixels, pixel2idx, boundary, boundaryPixels, offset)
    red_channel = laplacianSolver(background[:, :, 2], source[:, :, 2], 
                                    mask, maskPixels, pixel2idx, boundary, boundaryPixels, offset)
    
    image = np.stack([blue_channel, green_channel, red_channel], axis = 2)
    cv2.imwrite(result_path, image)
    
    cv2.imshow("result view", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
