# Scene Recognition

In this assignment, we utilize bag of words to do train a system to do scene recognition. This is essentially a pre-CNN way to classify images. We then utilize K-nearest neighbors to then classify new images. We first use a set of convolution filters to get multiple filter responses for each image. We then use a Harris corner detection algorithm to find the key points in an image, and we use the corresponding pixels found in the original and filter response to create our "bag of words". This set of words is then used in a knn algorithm to classify input images 

#### example image and its filter responses  
<img width="275" alt="Screenshot 2024-06-21 at 1 07 25 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/4691c51f-34af-4680-9e23-79f82c51f4ea">
<img width="282" alt="Screenshot 2024-06-21 at 1 07 32 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/1442f06f-322d-484d-8e58-23f357cd068a">
<img width="276" alt="Screenshot 2024-06-21 at 1 07 39 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/60c4571a-980c-4963-ace0-0fd9e0540b25">

#### Harris Corner detectio algorithm on multiple images 

<img width="277" alt="Screenshot 2024-06-21 at 1 08 46 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/1ec12f33-ba09-4a2c-bc9c-00328de5b6e0">
<img width="282" alt="Screenshot 2024-06-21 at 1 08 54 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/f5a67055-b463-4410-bfd8-133d3c9143ac">

#### final output of KNN with an accuracy of about 60%
<img width="431" alt="Screenshot 2024-06-21 at 1 09 48 PM" src="https://github.com/Jixi123/Computer_Vision/assets/86895390/51d679db-82fe-41a3-9887-742b852f11f9">
