import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

#Load data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

#General info training features
X_train.shape #(60000, 28, 28) three-dimensional matrix 28x28 (one training image) x 60000 (training images)
np.unique(Y_train) #0..255

#General info training output
Y_train.shape #(60000,)
np.unique(Y_train) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8) => multi-class classification

#General info testing features
X_test.shape #(10000, 28, 28) three-dimensional matrix 28x28 (one testing image) x 10000 (testing images)
Y_test.shape #(10000,)

#Draw first image and label
plt.imshow(X_train[0], cmap='gray') #two-dimensional matrix 28x28
title = "Label = " + str(Y_train[0])
plt.title(title)
plt.show()

'''
Grayscale images of 28x28 pixels (every pixel represented by a value 0..255) which representes as integer from 0 to 9
Note: if I had an rgb image each pixel would have been represented by an array of 3 values 0..255
'''

'''
Sequential is a plain stack of layers, every layer has 1 input tensor and 1 output tensor

ReLU activation function: returns input value, or 0 if the input is <= 0
Softmax activation function: to normalize the output to a probability distribution 
'''

model_1 = keras.Sequential()
model_1.add(keras.layers.Flatten(input_shape=(28,28,1), name='input')) #Input layer: to flatten from 28x28 to 784 (ANN requirement)
model_1.add(keras.layers.Dense(100, activation='relu', name='dense')) #Inner layer: to densely connect the previous state to this one
model_1.add(keras.layers.Dropout(0.5, name='drop')) #Inner layer: to prevent overfitting by turn off half of the neurons in the previous layer
model_1.add(keras.layers.Dense(10, activation='softmax', name='output')) #Output layer: 10 neurons whose value indicates the probability of belonging to one of the 10 classes

model_1.compile(
  loss = 'sparse_categorical_crossentropy', #which metric to use to optimally minimize the error
  metrics = ['accuracy'] #what metrics to monitor during learning (for user)
)

print("\n------SUMMARY MODEL 1------")
model_1.summary() #Trainable params: 79,510

print("\n------TRAINING MODEL 1------")
report = model_1.fit(
  X_train,
  Y_train,
  batch_size=10,
  epochs = 2 #every single epoch will run num_images/batch_size batches of batch_size images each (es. 60k/10images => 6000 batches for single epoch)
)

print("\n------TESTING MODEL 1------")
model_1.evaluate(#loss: 0.8178 - accuracy: 0.8636
  X_test, 
  Y_test,
  batch_size=10
) 

print("\n------PREDICTION MODEL 1------")
plt.imshow(X_test[1], cmap='gray') #draw number "2"
plt.show()
Y_test_predicted = model_1.predict(
  X_test,
  batch_size=10
)
Y_test_predicted[1] #array([3.2925278e-27, 1.5366599e-15, 9.9999994e-01, 6.9738913e-14,2.8905640e-36, 7.7120151e-27, 3.7774132e-17, 1.9843037e-23,1.1103685e-27, 0.0000000e+00])

#model_1.layers
#model_1.weights
#model_1.get_config()
#layer_input = model_1.get_layer('input')

#Save model to a file
model_1.save_weights(
  filepath='model/sequential',
  save_format='tf'
)

#Load weights (then learning) from model_1 saved to new model_2
model_2 = keras.Sequential()
model_2.add(keras.layers.Flatten(input_shape=(28,28,1), name='input')) 
model_2.add(keras.layers.Dense(100, activation='relu', name='dense')) 
model_2.add(keras.layers.Dropout(0.5, name='drop')) 
model_2.add(keras.layers.Dense(10, activation='softmax', name='output')) 

#Load weights 
model_2.load_weights(
  filepath='model/sequential'
)

#Now we decide to re-train only the last two layers
for layer in model_2.layers[:-1]:
  layer.trainable = False

model_2.compile(
  loss = 'sparse_categorical_crossentropy', 
  metrics = ['accuracy'] 
)

print("\n------SUMMARY MODEL 2------")
model_2.summary() #Trainable params: 78,500

print("\n------TRAINING LAST TWO LAYERS OF MODEL 2------")
report = model_2.fit(
  X_train,
  Y_train,
  batch_size=10,
  epochs = 2 
)

print("\n------TESTING MODEL 2------")
model_2.evaluate(#loss: 0.8178 - accuracy: 0.8636
  X_test, 
  Y_test,
  batch_size=10
)

print("\n------PREDICTION MODEL 2------")
plt.imshow(X_test[2], cmap='gray') #draw number "2"
plt.show()
Y_test_predicted = model_1.predict(
  X_test,
  batch_size=10
)
Y_test_predicted[2] 