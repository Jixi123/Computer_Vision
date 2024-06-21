import numpy as np
from getImageFeatures import get_image_features
import pickle 
from createFilterBank import create_filterbank


# -----fill in your implementation here --------
with open("../data/traintest.pkl", "rb") as x:
    traintest = pickle.load(x)
    
with open("./dictionaryHarris.pkl", "rb") as y:
    harris_dict = pickle.load(y)
    
with open("./dictionaryRandom.pkl", "rb") as z:
    random_dict = pickle.load(z)


random_file = open("./visionRandom.pkl", "wb")
harris_file = open("./visionHarris.pkl", "wb")

train_imagenames = traintest['train_imagenames']
train_labels = traintest['train_labels']

filter_bank = create_filterbank()

harris_vision_features = np.empty((1331, 500))
random_vision_features = np.empty((1331, 500))
for i, path in enumerate(train_imagenames):
    print('-- processing %d/%d' % (i, len(train_imagenames)))
    harris_wordmap = pickle.load(open('../data/%s_Harris.pkl' % path[:-4], 'rb'))
    random_wordmap = pickle.load(open('../data/%s_Random.pkl' % path[:-4], 'rb'))
    harris_vision_features[i] = get_image_features(harris_wordmap, len(harris_dict))
    random_vision_features[i] = get_image_features(random_wordmap, len(random_dict))    

vision_rand = {'dictionary': random_dict, 'filterBank': filter_bank,  'trainFeatures': random_vision_features, 'trainLabels': train_labels}
harris_rand = {'dictionary': harris_dict, 'filterBank': filter_bank,  'trainFeatures': harris_vision_features, 'trainLabels': train_labels}

pickle.dump(vision_rand, random_file)
pickle.dump(harris_rand, harris_file)

random_file.close()
harris_file.close()




# ----------------------------------------------
