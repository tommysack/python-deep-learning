import pandas as pd
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Softmax, Flatten, Dropout
import matplotlib.pyplot as plt

#Load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#General info
X_train.shape #(60000, 28, 28) three-dimensional matrix 28x28 (one training image) x 60000 (training images)
np.unique(Y_train) #0..255
Y_train.shape #(60000,)
np.unique(Y_train) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8) => multi-class classification
X_test.shape #(10000, 28, 28) three-dimensional matrix 28x28 (one testing image) x 10000 (testing images)
Y_test.shape #(10000,)

plt.imshow(X_train[0], cmap='gray') #two-dimensional matrix 28x28
title = "Label = " + str(Y_train[0])
plt.title(title)
plt.show()

'''
Grayscale images of 28x28 pixels (every pixel represented by a value 0..255) which representes as integer from 0 to 9
Note: if I had an rgb image each pixel would have been represented by an array of 3 values 0..255
'''

model = Sequential()
model.add(Flatten(input_shape=(28,28,1))) #Input layer: to flatten from 28x28 to 784 (ANN requirement)
model.add(Dense(100, activation='relu')) #Inner layer: to densely connect the previous state to this one
model.add(Dropout(0.5)) #Inner layer: to prevent overfitting by turn off half of the neurons in the previous layer
model.add(Dense(10, activation='softmax')) #Output layer: 10 neurons whose value indicates the probability of belonging to one of the 10 classes

model.compile(
  loss = 'sparse_categorical_crossentropy', #which metric to use to optimally minimize the error
  metrics = ['accuracy'] #what metrics to monitor during learning (for user)
)

print("\n------SUMMARY------")
model.summary() #Total params: 79,510

print("\n------TRAINING------")
report = model.fit(
  X_train,
  Y_train,
  batch_size=10,
  epochs = 2 #every single epoch will run num_images/batch_size batches of batch_size images each (es. 60k/10images => 6000 batches for single epoch)
)

print("\n------TESTING------")
model.evaluate(#loss: 0.8178 - accuracy: 0.8636
  X_test, 
  Y_test,
  batch_size=10
) 

print("\n------PREDICTION------")
plt.imshow(X_test[1], cmap='gray') #draw 2
plt.show()
Y_test_predicted = model.predict(
  X_test,
  batch_size=10
)
Y_test_predicted[1] #array([3.2925278e-27, 1.5366599e-15, 9.9999994e-01, 6.9738913e-14,2.8905640e-36, 7.7120151e-27, 3.7774132e-17, 1.9843037e-23,1.1103685e-27, 0.0000000e+00])



