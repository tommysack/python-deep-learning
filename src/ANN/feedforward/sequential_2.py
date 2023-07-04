import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

#Load data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

#Load weights (then learning) from model of sequential_1 saved to new model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28,1), name='input')) 
model.add(keras.layers.Dense(100, activation='relu', name='dense')) 
model.add(keras.layers.Dropout(0.5, name='drop')) 
model.add(keras.layers.Dense(10, activation='softmax', name='output')) 

#Load weights 
model.load_weights(
  filepath='model/sequential'
)

#Now we decide to re-train only the last two layers
for layer in model.layers[:-1]:
  layer.trainable = False

model.compile(
  loss = 'sparse_categorical_crossentropy', 
  metrics = ['accuracy'] 
)

print("\n------SUMMARY------")
model.summary() #Trainable params: 78,500

print("\n------TRAINING LAST TWO LAYERS------")
report = model.fit(
  X_train,
  Y_train,
  batch_size=10,
  epochs = 1
)

print("\n------TESTING------")
model.evaluate(#loss: 0.8178 - accuracy: 0.8636
  X_test, 
  Y_test,
  batch_size=10
)

print("\n------PREDICTION------")
plt.imshow(X_test[2], cmap='gray') #draw number "1"
plt.show()
Y_test_predicted = model.predict(
  X_test,
  batch_size=10
)
Y_test_predicted[2] 