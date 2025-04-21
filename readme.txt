
General Information
------------------

### Title:

-CNN Project 2 For Detecting AI in Images and Video

### Author:

-MB

### Prerequisite Libraries
-Pickle - module used to save models with a name
-OpenCV2 - pip install opencv-python - module used to interact with resizing test images, and other manipulative functions
-TensorFlow - pip install tensorflow - Nueral Network Library
-NumPy - pip install numpy - used to add dimensions to image arrays for testing 
-Matplot - pip install matplotlib - chart performance of CNN
-Dataset - https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset


### General Use
This program utilizes TensorFlow and other libraries to build, train, and test a CNN that can be used to make an average lump prediction based on testing data on whether or not
a photo or video has been manipulated or generated with AI. It utilizes a basic binary classification approach to do so. Provided is an example test set, an example trained ML model created with this program,
and methods to train more models off of the code.

To Train:
Once downloaded, Locate and replace the lines on 47, and 58. Replace these with the direct filepath to the training dataset example provided, or to your custom dataset.
After that is completed, go to line 168 for an example use case in training. uncomment the line and change the number of epochs you want to train your model for. Once this is done, run the program. 
Once this function is complete, it will save a pkl model to the active directory the program is running from

To Test:
change the "real_img_path" and "fake_img_path" variables to point to your testing data. Comment the train line and uncomment the test line. If you optionally chose to save your model as a separate name, you will have to change the name of the model you have saved.









