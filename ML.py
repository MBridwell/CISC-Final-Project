import tensorflow as tf
import numpy as np
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import cv2 # type:ignore
import os

#height and width of which the pictures get formatted to
img_height = 256
img_width = 256

#amount of images to be tested for each pass during training and validation
batch_size = 20

#actual cnn model -> taken from example, will need worked on to be more effecient
CNNModel = Sequential([
    layers.Input((256, 256, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(1, activation = 'sigmoid'),
])



#import training set from kaggle
# Import custom dataset
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    # Folder (training dataset)
    r'C:\Users\mason\Desktop\Dataset\dataset\dataset\test',  
    labels='inferred',
    label_mode="int",
    color_mode='grayscale',
    batch_size=batch_size,
    class_names=['0', '1'],  # Fixed comma
    image_size=(img_height, img_width),
    shuffle=True,
)
  
print(ds_train)

#import validation set from kaggle
ds_validate = tf.keras.preprocessing.image_dataset_from_directory(
    # Folder (validation dataset)
    r'C:\Users\mason\Desktop\Dataset\dataset\dataset\validate',  
    labels='inferred',
    label_mode="int",
    color_mode='grayscale',
    batch_size=batch_size,
    class_names=['0', '1'], 
    image_size=(img_height, img_width),
    shuffle=True,
    
)
print(ds_validate)


#compile the model. Loss and optimizer will need to be changed to improve results
CNNModel.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  
                  metrics=['accuracy'])

#print the history of the model
history = CNNModel.fit(ds_train, epochs=1, 
                    validation_data=(ds_validate))


#TODO:

 #   Create directory of images to be tested splicer.py can do this, but needs work
 #   Should return a series of images that are a specific resolution that have faces detected with opencv2
 #   pseudocode: 
 #  For i in test set
#   get image to test
# perform processing on image
# predict from trained model (#print(CNNModel.predict(imagedata)
# if model thinks it is fake, add fake counter
# if model thinks it is real, add real counter
# calculate percentage certainty off of counters, return that value as "final guess" (whichever counter is greater, percentage certainty)
# Opencv2 in theory isn't entirely 100% helpful to identify faces that are ai generated, so there should be a percentage loss as well incase there are false positives.
# How can i pull faces another way?
