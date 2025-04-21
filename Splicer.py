import cv2
import os


#file to be spliced for dataset generation
new_file = "C:\\Users\\mason\\Desktop\\Project1Temp\\"
os.mkdir(new_file)


#def for splitting video
def vidsplit():
    #save videocapture to a variable -> string of filename
    vidcap = cv2.VideoCapture('TestTwo.mkv')
    #float to take a capture once every x miliseconds, dont want this to be low as to not explode hard drive
    thirtyfps = float(1)
    #count var
    count = 0
    #read image from vidcap
    success,image = vidcap.read()

    #while successfully reading images
    while success:
        #change working directory to new file var
        os.chdir(new_file)
        
        cv2.imwrite("input_image_set_1%d.jpg" % count, image)     # save frame as JPEG file    TODO: needs Labeling for fake data, or real data
           
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count * thirtyfps))    # move the time 500 miliseconds in video
        #read next image
        success,image = vidcap.read()
        #successfully read new frame
        print('Read a new frame: ', success)
        #increase count
        count +=1
        #release video
    vidcap.release() 

def frecognition(testpicture):
    #change working directory to new file var
    os.chdir(new_file)

    imagePath = testpicture #image loaded -> load pciture to be tested
    img = cv2.imread(imagePath) #reading image
    

    
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts image to greyscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)  # Actual checking
    print (faces)
    

    if len(faces) == 0:  # No faces detected
        print(f"There is no face in image {testpicture}")
        os.remove(testpicture)
        return
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 112), 4)

        cropped_image = img[y:y+h, x:x+w]
        output_path = f"cropped_{testpicture}"  # Define the path to save cropped image
        cv2.imwrite(output_path, cropped_image)  # Save the cropped image

    # Save the image with bounding boxes 
    

    

    print(faces)
    if isinstance(faces, tuple): #face values are not stored as a tuple -> if tuple, no face, remove it
        
        print("There is no face in image ", testpicture)
        os.remove(testpicture)
    


def main():

    #split frames in video
    vidsplit()
    #change working directory
    os.chdir(new_file)
    #store directory in var directory
    directory = new_file
    

    #iterate throughout all split frames, if no face, remove them
    for filename in os.listdir(directory):
        print(filename)
        frecognition(filename)
        

if __name__ == "__main__":
    main()







