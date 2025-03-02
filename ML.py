import pickle as  p
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import cv2 # type:ignore
import os
import datetime
import matplotlib.pyplot as plt

#height and width of which the pictures get formatted to
img_height = 128
img_width = 128

#amount of images to be tested for each pass during training and validation
batch_size = 40

#actual cnn model -> taken from example, will need worked on to be more effecient
CNNModel = Sequential([
    layers.Input((img_height, img_width, 3)),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),

    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),

    layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)),

    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation = 'sigmoid'),

    
])


#datasets
#import training set from kaggle
# Import custom dataset
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    # Folder (training dataset)
    r'C:\Users\mason\Desktop\Dataset\dataset\dataset\train',  
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    class_names=['0', '1'], 
    image_size=(img_height, img_width),
    shuffle=True,
)

#import validation set from kaggle
ds_validate = tf.keras.preprocessing.image_dataset_from_directory(
    # Folder (validation dataset)
    r'C:\Users\mason\Desktop\Dataset\dataset\dataset\validate',  
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    class_names=['0', '1'], 
    image_size=(img_height, img_width),
    shuffle=True,
    
)

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    r'C:\Users\mason\Desktop\Dataset\dataset\dataset\test',  
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    class_names=['0', '1'], 
    image_size=(img_height, img_width),
    shuffle=False,
)



print(ds_validate)


##compile the model. Loss and optimizer will need to be changed to improve results
def compile():
    CNNModel.compile(optimizer='Adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  
                      metrics=[
                        tf.keras.metrics.FalseNegatives(),
                        tf.keras.metrics.FalsePositives(),
                        tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.TrueNegatives(),
                        'accuracy'
                        ])
    #
    #print the history of the model
    history = CNNModel.fit(ds_train, epochs=5, 
                        validation_data=(ds_validate))
    #
    ## save a model as pkl file
    with open('cnn_model_revision_6.pkl', 'wb') as f:
       p.dump(CNNModel, f)


    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.plot(history.history['false_negatives'], label = 'False negatives')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    #
    #
    print("Model saved as:  cnn_model_revision_6")
    

def test(model_name):
#to load and test model after:
    with open(model_name, 'rb') as f:
        loaded_model = p.load(f)

   #test_loss, false_negatives, test_accuracy, false_positives = loaded_model.evaluate(ds_test)

   #print("Test Loss: ", test_loss)
   #print("Test Accuracy: ", test_accuracy)
   #print("False Negatives: ", false_negatives)
   #print("False Positives: ", false_positives)

    image_path = 'Test2.png'

    image = cv2.imread(image_path)
    resized = cv2.resize(image, (128,128))
    image_with_channel = np.expand_dims(resized, axis=-1)
    image_dim = np.expand_dims(image_with_channel, axis=0)
    prediction = loaded_model.predict(image_dim)
    print(prediction)



    
   # ####Load the model
   # print("Model Loaded!")
   # ####test image path
   # image_path = 'Image.png'
   # ####read the image
   # image = cv2.imread(image_path)
   # ####resize the image
   # resized_image = cv2.resize(image,(512, 512))
   # ####normalize the image
   # normalize = resized_image / 255.0
   # ####add color channel
   # image_with_channel = np.expand_dims(normalize, axis=-1)
   # ####add tensor size
   # image_input = np.expand_dims(image_with_channel, axis=0)
   # ##
   # ####print the prediction (Unsure if working)
   # prediction = loaded_model.predict(image_input)
#
   # if prediction[0] >=0.5:
   #     print("Model predicts image to be fake")
   # else:
   #     print("Model predicts image to be real")


    #print("Raw Prediction:", prediction)
#
# compile()
test('cnn_model_revision_6.pkl')