import pickle as  p
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import cv2 # type:ignore
import os
import matplotlib.pyplot as plt
import statistics

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
# import custom train
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    r'filepath_here',  
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    class_names=['0', '1'], 
    image_size=(img_height, img_width),
    shuffle=True,
)

#import validation set from kaggle
#import custom validate
ds_validate = tf.keras.preprocessing.image_dataset_from_directory(
    r'filepath_here',  
    labels='inferred',
    label_mode="int",
    color_mode='rgb',
    batch_size=batch_size,
    class_names=['0', '1'], 
    image_size=(img_height, img_width),
    shuffle=True,
    
)




print(ds_validate)


#compile the model. Loss and optimizer will need to be changed to improve results
def compile(num_epochs):
    CNNModel.compile(optimizer='Adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  
                      metrics=[
                        tf.keras.metrics.FalseNegatives(),
                        tf.keras.metrics.FalsePositives(),
                        tf.keras.metrics.TruePositives(),
                        tf.keras.metrics.TrueNegatives(),
                        'accuracy'
                        ])
    
    #print the history of the model
    history = CNNModel.fit(ds_train, epochs=num_epochs, 
                        validation_data=(ds_validate))
    
    # save a model as pkl file
    with open('saved_model_name', 'wb') as f:
       p.dump(CNNModel, f)


    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    
    

def test(model_name):
#open saved modfel
    with open(model_name, 'rb') as f:
        loaded_model = p.load(f)

   #code for testing data file paths
    real_img_path = 'path_to_testing_data_real'
    fake_img_path = 'path_to_testing_data_fake'
    
   #make lists to store float values for averages later 
    real_pred = []
    fake_pred = []
    


    #for i in each path
        #image to test = i
        #preprocess i,
        #predict
        #add predict to list
        #return average of each list.

    for entry in os.scandir(real_img_path):  
        entry = preprocess(entry)
        prediction = loaded_model.predict(entry)
        #generic debug statement
        #print("Current Prediction: ", prediction)
        real_pred.append(float(prediction))

    for entry in os.scandir(fake_img_path):
        entry = preprocess(entry)
        prediction = loaded_model.predict(entry)
        #generic debug statement
        #print("Current Prediction: ", prediction)
        fake_pred.append(float(prediction))


    print("Final Guess of Real Dataset: ", statistics.mean(real_pred))
    print("Final Guess of Fake Dataset: ", statistics.mean(fake_pred))
    
    
    


def preprocess(entry):
    #read img
    image = cv2.imread(entry)
    #resize
    resized = cv2.resize(image, (128,128))
    #add additional channels
    image_with_channel = np.expand_dims(resized, axis=-1)
    image_dim = np.expand_dims(image_with_channel, axis=0)
    #return image_dim as output
    return image_dim



#Example function calls to run program
#compile(10)
#test('model_name')
