import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input,decode_predictions
from keras import utils 

try:

  # Load weights
  model_vgg16 = VGG16(
    weights='imagenet',
    classes=1000
  )

  # Load image
  image = utils.load_img(
    path='images/bike.jpg', 
    target_size=(224, 224) #The input size for VGG16 is 224
  )

  # Prepares to VGG16 format: converts to array and add one dimension, normalization, channel centering and blue channel filling
  image_array = utils.img_to_array(image) #(224, 224, 3)  
  image_array = np.expand_dims(image_array, axis=0) #(1, 224, 224, 3) 
  image_vgg16 = preprocess_input(image_array) 
  
  # Predictions
  classes_predictions = model_vgg16.predict(image_vgg16) #1000 classes 
  labels_predictions = decode_predictions(classes_predictions)

except Exception as e: 

  print("Exception occurred: " + str(e))

