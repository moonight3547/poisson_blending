import cv2
import numpy as np
from preprocess import preprocess, removeEdge
from poissonblending import poissonImageBlending

path = "imgs/"
id = "g"

#preprocess(path, id)
#removeEdge(path, id)

#poissonImageBlending(path, id, (10, 30)) #a
#poissonImageBlending(path, id, (20, 110)) #b
#poissonImageBlending(path, id, (220, 150)) #c
#poissonImageBlending(path, id, (0, 0)) #d
#poissonImageBlending(path, id, (130, 100)) #e
#poissonImageBlending(path, id, (150, 100)) #f
poissonImageBlending(path, id, (70, 50)) #g
#poissonImageBlending(path, id, (115, 35)) #h
