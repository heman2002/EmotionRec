
# coding: utf-8

# # Real time Prediction

import numpy as np
import cv2
from keras.preprocessing import image
import flask
from flask import Flask, request
from keras import backend as k
from keras.models import load_model
import pandas as pd
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.layers import Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import Adam

app = Flask(__name__)
app.config['upload_folder'] = './temp'

@app.route('/')
def index():
	return flask.jsonify('Hello, world')

@app.route('/emotion', methods=['POST'])
def upload():
	video = request.files['video']	#reads the video
	filename = secure_filename(video.filename) 
	video.save(os.path.join(app.config['upload_folder'], filename)) #saving video in temp file
	#print(filename)
	response = process_video(filename) #function call to process video
	k.clear_session() # clearing kera session after every api call
	return flask.jsonify(response)

def process_video(video):
	predicted_emotions = []
	face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml') # reading haar cascade xml file 
	#print(video)
	cap = cv2.VideoCapture('./temp/' + video) # capturing video in opencv
	#cap.open(os.path.join(app.config['upload_folder'], video))
	#print(cap.isOpened())
	model=load_model('emotion_model.h5') # loading stored model
	emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') # all emotions
	count = 0
	while(True):
		ret, img = cap.read() # reading frame by frame
		#print(ret)
		if ret is False:
			break
		
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting image to grayscale

		faces = face_cascade.detectMultiScale(gray, 1.3, 5) # detecting faces in image

		#print(faces) #locations of detected faces

		for (x,y,w,h) in faces:
			#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
		
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
			#preprocessing image
			detected_face = cv2.GaussianBlur(detected_face,(5,5),0) #smoothing image
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)) #CLAHE
			detected_face = clahe.apply(detected_face)
			#cv2.imshow('img', detected_face)

			img_pixels = image.img_to_array(detected_face) # converting to an array
			img_pixels = np.expand_dims(img_pixels, axis = 0) # increasing image dimensions
	    
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

			predictions = model.predict(img_pixels) #store probabilities of 7 expressions

			max_index = np.argmax(predictions[0]) # finding emotion with highest probability 

			emotion = emotions[max_index]
			predicted_emotions.append(emotion)
			#cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
		if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
			break
		

	#kill open cv things		
	cap.release()
	cv2.destroyAllWindows()
	#print(count)
	return predicted_emotions


@app.route('/model', methods=['POST'])
def model():
	lines, num_of_instances = read_kaggle_emotion_dataset() #call function to read kaggle dataset
	x_train, y_train, x_test, y_test = split_train_test(lines, num_of_instances) # split dataset into training and testing
	x_train, y_train, x_test, y_test = data_transformation(x_train, y_train, x_test, y_test) #normalising and reshaping data
	model = build_model() # building keras model
	model = train_model(x_train, y_train, batch_size, epochs) #training the model
	train_loss, train_accuracy, test_loss, test_accuracy = evaluate_model(x_train, y_train, x_test, y_test, model) #evaluating accuracy and loss
	response = 'Model trained successfully. \nTrain loss:' + train_loss + '\nTrain accuracy:' + train_accuracy + '\n Test loss:' + test_loss + '\n Test accuracy:' + test_accuracy 	
	return flask.jsonify(response)

def read_kaggle_emotion_dataset():
	#kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
	with open('fer2013.csv') as f:
		content = f.readlines() #reading the dataset

	lines = np.array(content) # converting data into numpy array
 
	num_of_instances = lines.size # calculating number of records
	#print("number of instances: ",num_of_instances)
	return (lines, num_of_instances)

def split_train_test(lines, num_of_instances):
	x_train, y_train, x_test, y_test = [], [], [], []
	num_classes = 7 
	for i in range(1,num_of_instances):
		try:
        		emotion, img, usage = lines[i].split(',') #splitting csv records into corresponding columns
   
        		val = img.split(' ')
        		pixels = np.array(val, 'float32')
 
        		emotion = keras.utils.to_categorical(emotion, num_classes) # converting to categorical y
 
        		if 'Training' in usage:
            			y_train.append(emotion)
            			x_train.append(pixels)
        		elif 'PublicTest' in usage:
            			y_test.append(emotion)
            			x_test.append(pixels)
    		except:
        		return 'Error with reading data'
	return (x_train, y_train, x_test, y_test)

def data_transformation(x_train, y_train, x_test, y_test):
	x_train = np.array(x_train, 'float32') # converting to numpy array
	y_train = np.array(y_train, 'float32')
	x_test = np.array(x_test, 'float32')
	y_test = np.array(y_test, 'float32')

	x_train /= 255 #normalize inputs between [0, 1]
	x_test /= 255

	x_train = x_train.reshape(x_train.shape[0], 48, 48, 1) #reshaping image to (48,48,1)
	x_train = x_train.astype('float32')
	x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
	x_test = x_test.astype('float32')

	return (x_train, y_train, x_test, y_test)

def build_model():
	model = Sequential() # sequential model
 
	#1st convolution layer
	model.add(Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
	 
	#2nd convolution layer
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
	 
	#3rd convolution layer
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
	 
	model.add(Flatten())
	 
	#fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	 
	model.add(Dense(num_classes, activation='softmax'))
	#model.summary()
	return model

def train_model(x_train, y_train, batch_size, epochs):
	gen = image.ImageDataGenerator()
	train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
	 
	model.compile(loss='categorical_crossentropy' 
	, optimizer=keras.optimizers.Adam()
	, metrics=['accuracy']
	) #compiling model
 
	model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs) #fitting data

	model.save('emotion_model.h5')

	return model

def evaluate_model(x_train, y_train, x_test, y_test, model):
	train_score = model.evaluate(x_train, y_train, verbose=0) #evaluating accuracy and loss
	#print('Train loss:', train_score[0])
	train_loss = train_score[0]
	#print('Train accuracy:', 100*train_score[1])
	train_accuracy = 100*train_score[1]
	 
	test_score = model.evaluate(x_test, y_test, verbose=0)
	#print('Test loss:', test_score[0])
	test_loss = test_score[0]
	#print('Test accuracy:', 100*test_score[1])
	test_accuracy = 100*test_score[1]

	return (train_loss, train_accuracy, test_loss, test_accuracy)

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
	return """Wrong URL!<pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
	return """An internal error occurred: <pre>{}</pre>See logs for full stacktrace.""".format(e), 500
    
if __name__ == '__main__':
	app.run('localhost', 8090, debug=True)
