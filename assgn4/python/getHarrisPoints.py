import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    x = ndimage.sobel(I,0)
    y = ndimage.sobel(I,1)

    neighbor = np.ones((3,3))
    
    top_left = ndimage.convolve(x*x, neighbor)
    top_right, bottom_left = ndimage.convolve(x*y, neighbor), ndimage.convolve(x*y, neighbor)
    bottom_right = ndimage.convolve(y*y, neighbor)

    det = top_left * bottom_right - top_right*bottom_left
    R = det - k*((top_left+bottom_right)**2)
    
    temp_points = np.argpartition(R.ravel(), -alpha)[-alpha:]
    y_coords, x_coords = np.unravel_index(temp_points, R.shape)
    
    points = np.transpose(np.vstack((np.array(x_coords), np.array(y_coords))))
 
    # ----------------------------------------------
    
    return points

