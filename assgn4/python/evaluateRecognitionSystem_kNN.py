import numpy as np
from getImageFeatures import get_image_features
import pickle 
from getImageDistance import get_image_distance
from sklearn import metrics

# -----fill in your implementation here --------

with open("../data/traintest.pkl", "rb") as x:
    traintest = pickle.load(x)
    
with open("./visionHarris.pkl", "rb") as z:
    visionHarris = pickle.load(z)
    
harris_dict = visionHarris['dictionary']
harris_bank = visionHarris['filterBank']
harris_features = visionHarris['trainFeatures']
harris_labels = visionHarris['trainLabels']
test_imagenames = traintest['test_imagenames']
test_labels = traintest['test_labels']

harris_chi_matrix = []
harris_chi_metric = 0
best_k = 0
for k in range(1, 41):
    harris_chi_guesses = []
    for i, path in enumerate(test_imagenames):
        print('-- processing %d/%d' % (i, len(test_imagenames)))
        harris_wordmap = pickle.load(open('../data/%s_Harris.pkl' % path[:-4], 'rb'))
        harris_vision_feature = get_image_features(harris_wordmap, len(harris_dict))
        hc = []
        for j in range(len(harris_features)):
            hc = np.append(hc, get_image_distance(harris_vision_feature, harris_features[j].reshape(1,-1), 'chi2'))
        arr = np.argpartition(hc, k)[:k]
        val = []
        for j in arr:
            val = np.append(val, harris_labels[j])
        val = val.astype(int)
        harris_chi_guesses = np.append(harris_chi_guesses, np.bincount(val).argmax())
    if(metrics.accuracy_score(harris_chi_guesses, test_labels)>harris_chi_metric):
        harris_chi_metric = metrics.accuracy_score(harris_chi_guesses, test_labels)
        harris_chi_matrix = metrics.confusion_matrix(harris_chi_guesses, test_labels)
        best_k = k

print("\nharris chi matrix: \n", harris_chi_matrix)
print("harris chi metric:", harris_chi_metric)
print("best k:", k)

        
    
# ----------------------------------------------
