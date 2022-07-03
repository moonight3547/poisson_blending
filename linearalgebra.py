import cv2
import numpy as np

# A function to return the guide vector field v_pq
def guideVec(pixel : tuple, 
            neighbor : tuple, 
            bkg : np.ndarray,
            src : np.ndarray,
            offset : tuple,
            mixed_grad : bool = False
    ) -> int:
    offset_y, offset_x = offset
    pixel_y, pixel_x = pixel
    neigh_y, neigh_x = neighbor
    src_pixel = (pixel_y-offset_y+1, pixel_x-offset_x+1)
    src_neigh = (neigh_y-offset_y+1, neigh_x-offset_x+1)
    if not mixed_grad :
        try:
            return int(src[src_pixel]) - int(src[src_neigh])
        except Exception as e:
            print("error", e)
            return 0
    else:
        diff_bkg = int(bkg[pixel]) - int(bkg[neighbor])
        try:
            diff_src = int(src[src_pixel]) - int(src[src_neigh])
            if abs(diff_bkg) > abs(diff_src) :
                return diff_bkg
            else :
                return diff_src
        except :
            return diff_bkg
