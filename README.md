# EmotionRec
## About Project:
The project emotion recognition is built to predict the emotions of people real time.

## Dataset:
Kaggle Emotion Recogntion dataset, kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge

## Libraries used:
OpenCV: Primarily used for performing image processing and face detection
Keras: Deep learning CNNs were built using keras with tensorflow as backend 
Flask: The api was built using it.

## Api end points:

### 1.'/emotion'

#### Functions:

upload()
Reading the video and calling process_video() function

process_video()
Read frontal face features for face detection
Perform face detection
For all frames a series of image processing steps are performed , i.e., Resizing, smoothing(Gaussian blurring), CLAHE(Histogram equalisation)
Loading model
Predicting emotion

### 2.'/model'

#### Functions:

model()
Calling functions to read dataset, split into training and testing, normalising and reshaping

read_kaggle_emotion_dataset()
Reading csv file containing image dataset

split_train_test()
Splitting data into training and testing set

data_transformation()
transforming data by normalising it and reshaping it

build_model()
giving CNN model architecture and building it

train_model()
fitting emotion dataset to CNN architecture 

evaluate_model()
evaluating accuracy and loss of the model for the dataset

