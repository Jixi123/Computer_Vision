import pickle
from getDictionary import get_dictionary


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------
a = 200
K = 500

random_file = open("./dictionaryRandom.pkl", "wb")
harris_file = open("./dictionaryHarris.pkl", "wb")

random = get_dictionary(train_imagenames, a, K, "Random")
harris = get_dictionary(train_imagenames, a, K, "Harris")

pickle.dump(random, random_file)
pickle.dump(harris, harris_file)

random_file.close()
harris_file.close()



# ----------------------------------------------



