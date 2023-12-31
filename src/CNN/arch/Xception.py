import numpy as np
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input,decode_predictions
from keras import utils 

try:

  # Load weights
  model_xception = Xception(
    weights='imagenet',
    classes=1000
  )

  # Load image and resize to dimension for Xception model
  image = utils.load_img(
    path='bike.jpg',
    target_size=(299,299)
  )

  # Prepares to Xception format: converts to array and add one dimension, then prepares the image for the Xception model 
  image_array = utils.img_to_array(image) #(224, 224, 3)  
  image_array = np.expand_dims(image_array, axis=0) #(1, 224, 224, 3) 
  image_xception = preprocess_input(image_array) 
  
  # Predictions
  classes_predictions = model_xception.predict(image_xception) #1000 classes 
  labels_predictions = decode_predictions(classes_predictions)

except Exception as e: 

  print("Exception occurred: " + str(e))

