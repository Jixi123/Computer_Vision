
import numpy as np
from utils import chi2dist
from scipy.spatial.distance import cdist

def get_image_distance(hist1, hist2, method):

    # -----fill in your implementation here --------
    if(method == 'euclidean'):
        dist = cdist(hist1, hist2, 'euclidean')
    else:
        dist = chi2dist(hist1, hist2)

    # ----------------------------------------------
    
    return dist
