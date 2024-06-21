import numpy as np



def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    wordMap = wordMap.flatten()
    h = np.bincount(wordMap, minlength = dictionarySize)
    h = (h / np.linalg.norm(h)).reshape(1, dictionarySize)
  
    # ----------------------------------------------
    
    return h
