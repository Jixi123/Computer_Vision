import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses


def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    responses = extract_filter_responses(I, filterBank)
    reshaped_response = np.reshape(responses, (responses.shape[0]*responses.shape[1],60))
    errors = np.argmin(cdist(reshaped_response, dictionary, "euclidean"), -1)
    wordMap = np.reshape(errors, (responses.shape[0], responses.shape[1]))
    
    # ----------------------------------------------

    return wordMap

