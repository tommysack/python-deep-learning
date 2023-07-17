import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

#Load data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

#General info training features
X_train.shape #(60000, 28, 28) three-dimensional matrix 28x28 (one training image) x 60000 (training images)
np.amin(X_train) #0
np.amax(X_train) #255

#General info training target
Y_train.shape #(60000,)
np.unique(Y_train) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8) => multi-class classification

#General info testing features
X_test.shape #(10000, 28, 28) three-dimensional matrix 28x28 (one testing image) x 10000 (testing images)
Y_test.shape #(10000,)

#Draw first image and label
plt.imshow(X_train[0], cmap='gray') 
title = "Label = " + str(Y_train[0])
plt.title(title)
plt.show()

#Check if X needs to scaling
print("\nBEFORE scaling")
print("X train min", np.amin(X_train))
print("X test min", np.amin(X_test))
print("X train max", np.amax(X_train))
print("X test max", np.amax(X_test))

X_train = X_train / 255
X_test = X_test / 255

print("\nAFTER scaling")
print("X train min", np.amin(X_train))
print("X test min", np.amin(X_test))
print("X train max", np.amax(X_train))
print("X test max", np.amax(X_test))

'''
Grayscale images of 28x28 pixels (every pixel represented by a value 0..255) which representes as integer from 0 to 9
Note: if I had an rgb image each pixel would have been represented by an array of 3 values 0..255
'''

'''
Convolutionary use convolutional layers (CNN), it exploits the structures present in visual data.
The layers apply a set of "kernels" (learnable filters), to the input data.
Sliding the filters over the input it computes a product between the filter and a small region of the input.

It captures local patterns and spatial dependencies in the input data. The first layers will learn low-level
features of image like textures, edges, .. while the last layers will learn the high-level structure of image.
It reduces the number of params to be learned because it shares weights across different locations of image.
'''

model = keras.Sequential()

# First convolutional block
model.add(keras.layers.Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1))) #To extract information from different areas of the images: 8 filters of size 3x3
model.add(keras.layers.BatchNormalization()) #To normalize
model.add(keras.layers.MaxPooling2D((2,2), padding='same')) #To gradually sacrifice spatial information for a progressively more abstract encoding of the images
model.add(keras.layers.Dropout(0.2)) #To prevent overfitting by turn off neurons in the previous layer

# Second convolutional block
model.add(keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')) #To extract information from different areas of the images: 4 filters of size 3x3
model.add(keras.layers.BatchNormalization()) #To normalize
model.add(keras.layers.MaxPooling2D((2,2), padding='same')) #To gradually sacrifice spatial information for a progressively more abstract encoding of the images
model.add(keras.layers.Dropout(0.4)) #To prevent overfitting by turn off neurons in the previous layer

# Flatten
model.add(keras.layers.Flatten()) #To flat from multidimensional matrices of numbers

# Output layer 
model.add(keras.layers.Dense(10, activation='softmax')) #10 neurons whose value indicates the probability of belonging to one of the 10 classes

model.compile(
  loss = 'sparse_categorical_crossentropy', #which metric to use to optimally minimize the error
  metrics = ['accuracy'] #what metrics to monitor during learning (for user)
)

print("\n------SUMMARY------")
model.summary() #Trainable params: 4,354

print("\n------TRAINING------")
report = model.fit( # loss: 0.3360 - accuracy: 0.8987
  X_train,
  Y_train,
  batch_size=10,
  epochs = 1 #every single epoch will run num_images/batch_size batches of batch_size images each (es. 60k/10images => 6000 batches for single epoch)
)

print("\n------TESTING------")
model.evaluate( # loss: 0.1038 - accuracy: 0.9683
  X_test, 
  Y_test,
  batch_size=10
) 

print("\n------PREDICTION------")
plt.imshow(X_test[1], cmap='gray') #draw number "2"
plt.show()
Y_test_predicted = model.predict(
  X_test,
  batch_size=10
)
Y_test_predicted[1] #array([4.73006412e-06, 7.38925928e-06, 9.99987602e-01, 1.32351774e-08, 1.30660575e-14, 1.93654415e-11, 3.00965652e-09, 2.54459100e-16, 2.27350512e-07, 8.08293929e-14])



