import cv2
import numpy as np

# A function to rescale input image, save rescaled image, and return the new dimension
def rescaleImage(in_name : str,       # filename(path) of input image
                 out_name : str,      # filename(path) of output image to save the rescaled image
                 out_size : tuple,    # (height, width), describing the dimension of the rescaled image
                 axis : int,         # -1 for free rescaling, 0/1 for equal-ratio rescaling depending on axis = 0/1
                ) -> None :
    assert(len(out_size) == 2)
    assert(axis == 0 or axis == 1 or axis == -1)
    
    in_img = cv2.imread(in_name)
    in_size = in_img.shape[0:2]
    
    if axis == -1:
        out_img = cv2.resize(in_img, out_size)
    else :
        ratio = out_size[axis] / in_size[axis]
        out_img = cv2.resize(in_img, None, fx = ratio, fy = ratio)
        out_size = out_img.shape[0:2]
    
    cv2.imwrite(out_name, out_img)
    return out_size

white_color = (255, 255, 255)
window_name = "mask generation"
temp_img = None
temp_mask = None
points = []

# the callback function during mouse event
def onMouse(event, x, y, flags, param):
    global temp_img
    global temp_mask
    global points

    # print(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        points = []
        start_point = (x, y)
        cv2.circle(temp_img, start_point, 1, white_color, 0)
        cv2.imshow(window_name, temp_img)
        points.append(start_point)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        current_point = (x, y)
        cv2.line(temp_img, points[-1], current_point, white_color, 1)
        cv2.imshow(window_name, temp_img)
        points.append(current_point)
        
    elif event == cv2.EVENT_LBUTTONUP:
        start_point = points[0]
        cv2.line(temp_img, points[-1], start_point, white_color, 1)
        cv2.imshow(window_name, temp_img)
        points.append(start_point)
        
        cv2.circle(temp_mask, start_point, 1, white_color, 0)
        sizePoints = len(points)
        for i in range(1, sizePoints):
            cv2.line(temp_mask, points[i-1], points[i], white_color, 1)
        
#        cv2.floodFill(mask, mask_img, (x, y), )
#        points = []



# A function to generate a mask upon input image
def generateMask(in_name : str) -> None :
    global temp_img
    global temp_mask
    in_img = cv2.imread(in_name)
    temp_img = in_img
    mask = np.zeros(in_img.shape, dtype = np.uint8)
    temp_mask = mask
    out_name = in_name.split('_')[0] + '_contour.jpg'
    mask_name = in_name.split('_')[0] + '_mask.jpg'
    msk_name = in_name.split('_')[0] + '_t1.jpg'
    img_name = in_name.split('_')[0] + '_t2.jpg'

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, onMouse)
    cv2.imshow(window_name, temp_img)
    
    while(True):
        c = cv2.waitKey(3000)&0xFF
        print("get command: ", c)
        if c==ord('s'): # save
            in_img = temp_img.copy()
            mask = temp_mask.copy()
            cv2.imshow(window_name, temp_img)
            cv2.imwrite(out_name, mask)
        elif c==ord('b'): # back
            temp_img = in_img.copy()
            temp_mask = mask.copy()
            cv2.imshow(window_name, temp_img)
        elif c==ord('q'): # quit
            contour = mask
            h, w = in_img.shape[:2]
            fillmask = np.zeros((h+2, w+2), dtype = np.uint8)
            x, y = points[0]
            while(mask[y][x][0] or mask[y][x][1] or mask[y][x][2]) :
                y -= 1
            cv2.floodFill(mask, fillmask, (x, y), white_color,(5, 5, 5), (5, 5, 5), cv2.FLOODFILL_FIXED_RANGE)
            cv2.imwrite(mask_name, mask)
            cv2.destroyAllWindows()
            break
