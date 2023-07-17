import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

#Load data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

#General info training features
X_train.shape #(50000, 32, 32, 3) three-dimensional matrix 32x32x3 (one training image) x 50000 (training images)
np.amin(X_train) #0
np.amax(X_train) #255

#General info training target
Y_train.shape #(50000, 1)
np.unique(Y_train) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8) => multi-class classification

#General info testing features
X_test.shape #(10000, 32, 32, 3) three-dimensional matrix 32x32x3 (one testing image) x 10000 (testing images)
Y_test.shape #(10000, 1)

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
'''

model = keras.Sequential()

# First convolutional block
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.1))

# Secondo convolutional block
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.2))

# Third convolutional block
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.3))

# Flatten
model.add(keras.layers.Flatten())

# Output layer
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
  loss = 'sparse_categorical_crossentropy', #which metric to use to optimally minimize the error
  metrics = ['accuracy'] #what metrics to monitor during learning (for user)
)

print("\n------SUMMARY------")
model.summary() #Trainable params: 325,866

print("\n------TRAINING------")
report = model.fit( # loss: 0.4624 - accuracy: 0.8392
  X_train,
  Y_train,
  batch_size=10,
  epochs = 10 #every single epoch will run num_images/batch_size batches of batch_size images each (es. 60k/10images => 6000 batches for single epoch)
)

print("\n------TESTING------")
model.evaluate( # loss: 0.5902 - accuracy: 0.8062
  X_test, 
  Y_test,
  batch_size=10
) 

