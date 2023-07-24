import tensorflow as tf
import keras

# Load data
(X_train, Y_train), (X_test, Y_test) = keras.datasets.imdb.load_data(num_words=10000) #vocabolary of 10000 words

# Limit the maximum length of the sequences (each reviews max 200 words) and perform padding if necessary
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=200)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=200)

model = keras.Sequential()

# Trasforms vocabolary (10000) in a low (50) dimensional space of integers (similar words are closer together)
# 200 is the maximum number of words for each review (features) 
model.add(keras.layers.Embedding(input_dim=10000, output_dim=50, input_shape=(200,))) 
model.add(keras.layers.SimpleRNN(64)) #Analyze the sequences of words and capture dependencies between them
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(
  loss='binary_crossentropy',
  metrics=['accuracy'],
  optimizer='adam'
)

print("\n------SUMMARY------")
model.summary() #Trainable params: 507,425

print("\n------TRAINING------")
model.fit(
  X_train, 
  Y_train, 
  batch_size=32,
  epochs=5
)

print("\n------TESTING------")
model.evaluate( # loss: 0.5901 - accuracy: 0.7182
  X_test,
  Y_test,
  batch_size=10
)

