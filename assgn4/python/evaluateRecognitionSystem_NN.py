import numpy as np
from getImageFeatures import get_image_features
import pickle 
from getImageDistance import get_image_distance
from sklearn import metrics

# -----fill in your implementation here --------

with open("../data/traintest.pkl", "rb") as x:
    traintest = pickle.load(x)

with open("./visionRandom.pkl", "rb") as y:
    visionRand = pickle.load(y)
    
with open("./visionHarris.pkl", "rb") as z:
    visionHarris = pickle.load(z)
    
harris_dict = visionHarris['dictionary']
random_dict = visionRand['dictionary']

harris_bank = visionHarris['filterBank']
random_bank = visionRand['filterBank']

harris_features = visionHarris['trainFeatures']
random_features = visionRand['trainFeatures']


harris_labels = visionHarris['trainLabels']
random_labels = visionRand['trainLabels']

test_imagenames = traintest['test_imagenames']
test_labels = traintest['test_labels']

rand_euclid_guesses = []
rand_chi_guesses = []
harris_euclid_guesses = []
harris_chi_guesses = []


for i, path in enumerate(test_imagenames):
    print('-- processing %d/%d' % (i, len(test_imagenames)))
    harris_wordmap = pickle.load(open('../data/%s_Harris.pkl' % path[:-4], 'rb'))
    random_wordmap = pickle.load(open('../data/%s_Random.pkl' % path[:-4], 'rb'))
    
    harris_vision_feature = get_image_features(harris_wordmap, len(harris_dict))
    random_vision_feature = get_image_features(random_wordmap, len(random_dict))
    
    he = []
    hc = []
    re = []
    rc = []
    
    for j in range(len(harris_features)):
        he = np.append(he, get_image_distance(harris_vision_feature, harris_features[j].reshape(1,-1), 'euclidean'))
        hc = np.append(hc, get_image_distance(harris_vision_feature, harris_features[j].reshape(1,-1), 'chi2'))
        re = np.append(re, get_image_distance(random_vision_feature, random_features[j].reshape(1,-1), 'euclidean'))
        rc = np.append(rc, get_image_distance(random_vision_feature, random_features[j].reshape(1,-1), 'chi2'))
        
    min_harris_euclid_val = harris_labels[np.argmin(he)]
    min_harris_chi_val = harris_labels[np.argmin(hc)]
    min_random_euclid_val = random_labels[np.argmin(re)]
    min_random_chi_val = random_labels[np.argmin(rc)]
         
    rand_euclid_guesses = np.append(rand_euclid_guesses, min_random_euclid_val)
    rand_chi_guesses = np.append(rand_chi_guesses, min_random_chi_val)
    harris_euclid_guesses = np.append(harris_euclid_guesses, min_harris_euclid_val)
    harris_chi_guesses = np.append(harris_chi_guesses, min_harris_chi_val)

rand_euclid_matrix = metrics.confusion_matrix(rand_euclid_guesses, test_labels)
rand_euclid_metric = metrics.accuracy_score(rand_euclid_guesses, test_labels)

rand_chi_matrix = metrics.confusion_matrix(rand_chi_guesses, test_labels)
rand_chi_metric = metrics.accuracy_score(rand_chi_guesses, test_labels)

harris_euclid_matrix = metrics.confusion_matrix(harris_euclid_guesses, test_labels)
harris_euclid_metric = metrics.accuracy_score(harris_euclid_guesses, test_labels)

harris_chi_matrix = metrics.confusion_matrix(harris_chi_guesses, test_labels)
harris_chi_metric = metrics.accuracy_score(harris_chi_guesses, test_labels)

print("\nrandom euclidean matrix: \n", rand_euclid_matrix)
print("random euclid metric:", rand_euclid_metric)

print("\nrandom chi matrix: \n", rand_chi_matrix)
print("random chi metric:", rand_chi_metric)

print("\nharris euclidean  matrix: \n", harris_euclid_matrix)
print("harris euclidean metric:", harris_euclid_metric)

print("\nharris chi matrix: \n", harris_chi_matrix)
print("harris chi metric:", harris_chi_metric)
# ----------------------------------------------
