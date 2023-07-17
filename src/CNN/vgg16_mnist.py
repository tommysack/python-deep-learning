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
np.unique(Y_train) #array([0, 1, 2, (3,3), 4, 5, 6, 7, 8, 9], dtype=uint8) => multi-class classification

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


model = keras.Sequential()

model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D((2,2), padding='same'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
  loss = 'sparse_categorical_crossentropy', #which metric to use to optimally minimize the error
  metrics = ['accuracy'] #what metrics to monitor during learning (for user)
)

print("\n------SUMMARY------")
model.summary() #Trainable params: 37,706,050

print("\n------TRAINING------")
report = model.fit( # loss: 0.2110 - accuracy: 0.9706
  X_train,
  Y_train,
  batch_size=10,
  epochs = 5 #every single epoch will run num_images/batch_size batches of batch_size images each (es. 60k/10images => 6000 batches for single epoch)
)

print("\n------TESTING------")
model.evaluate( # loss: 25.2325 - accuracy: 0.5900
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



