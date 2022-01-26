# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# !pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
# !pip install tensorflow 
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
#%matplotlib inline



## load the data
from keras.datasets import cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)


# image classification
classification = ['Airplane','Automobile','Bird','Cat', 'Deer','Dog','Frog','Horse','Ship','Truck']

# see the first image as an array
img_1 = 30
x_train[img_1]
# show the image as a picture
plt.imshow(x_train[img_1])

print("The image is a ", classification[y_train[img_1][0]])


# converting all the labells into a set of 10 numbers to input into the neural Network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)


# print the one hot label of the above image
print('The one hot label is ', y_train_one_hot[img_1])


x_train = x_train/255
x_test = x_test/255


x_train[img_1]




model = Sequential()

# Add first layer of the cnn
model.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (32,32,3)))

# Add a max pooling layer
model.add(MaxPooling2D(pool_size =(2,2)))


# Add a second convolution layeer
model.add(Conv2D(32, (5,5), activation = 'relu'))

# Add another max pooling layer
model.add(MaxPooling2D(pool_size =(2,2)))

# Add a flattening layer
model.add(Flatten())

# add a layer with 5000 neurons
model.add(Dense(5000, activation = 'relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# add a layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# add a layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

# Add a drop out layer
model.add(Dropout(0.5))

# add a layer with 250 neurons
model.add(Dense(250, activation = 'relu'))

# add a layer with 10 neurons
model.add(Dense(10, activation = 'softmax'))




model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])


# # Implement callback function to stop training  when accuracy reaches ACCURACY_THRESHOLD
# ACCURACY_THRESHOLD = 0.95

# class myCallback(tf.keras.callbacks.Callback):
# 	def on_epoch_end(self, epoch, logs={}):
# 		if(logs.get('acc') > ACCURACY_THRESHOLD):
# 			print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
# 			self.model.stop_training = True

# # Instantiate a callback object
# callbacks = myCallback()

hist = model.fit(x_train, y_train_one_hot,
                batch_size = 100,
                epochs = 3,
                validation_split = 0.2)
                #  callbacks=[callbacks])
                
                

model.evaluate(x_test, y_test_one_hot)[1]



plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc ='upper left')
plt.show()


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc ='upper right')
plt.show()



import os
# !pip install opencv-python

import cv2
# !pip install scikit-image
from skimage.transform import resize

foldername = "./test_images/"

# my_list = []
# for img_count, filename in enumerate(os.listdir(foldername)):
# img = plt.imread('/content/test_images/download (3).jpeg')
img = plt.imread(os.path.join(foldername,'download4.jpg'))
plt.imshow(img)
resized_image = resize(img, (32,32, 3))
     
# for item in my_list:
predictions = model.predict(np.array(resized_image))
print(predictions)
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

################ display the array as integers #########
for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            #swap the numbers
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp
display(list_index)
        
for i in range(5):
    print(classification[list_index[i]], ':', round((predictions[0][list_index[i]] * 100),2), '%')
print("=" * 50)

# print('Total images are :', img_count+1)

