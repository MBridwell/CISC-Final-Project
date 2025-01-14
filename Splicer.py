import cv2
import os

new_file = "C:\\Users\\Mason\\Desktop\\Project1Temp\\Example"
os.mkdir(new_file)



def vidsplit():
    
    vidcap = cv2.VideoCapture('test.mp4')
    thirtyfps = float(500)
    count = 0
    success,image = vidcap.read()

    while success:
        os.chdir(new_file)
        cv2.imwrite("input_image_%d .jpg" % count, image)     # save frame as JPEG file      
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*thirtyfps))    # move the time
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count +=1
    vidcap.release() 

def frecognition(testpicture):
    os.chdir(new_file)

    imagePath = testpicture #image loaded
    img = cv2.imread(imagePath) #reading image
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts image to greyscale

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    face = face_cascade.detectMultiScale(gray_image, 1.3, 5) #actual checking 

    print(face)
    if isinstance(face, tuple):
        
        print("There is no face in image ", testpicture)
        os.remove(testpicture)

        
        
   # for (x, y, w, h) in face: #draws rectangle around face given in coordinates.
     #   cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 112), 4)

    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #cv2.imshow('img',img) #shows image.

    #k = cv2.waitKey(0)
    #if k == 27:         # wait for ESC key to exit
     #   cv2.destroyAllWindows()
    #elif k == ord('s'): # wait for 's' key to save and exit
     #   cv2.imwrite('messigray.png',img)
      #  cv2.destroyAllWindows()

#vidsplit()
#frecognition(parameterLOL)
#os.chdir(new_file)
#directory = new_file
    
#for filename in os.listdir(directory):
    #print(filename)
 #   frecognition(filename)

def main():
    vidsplit()

    os.chdir(new_file)

    directory = new_file
    
    for filename in os.listdir(directory):
        print(filename)
        frecognition(filename)

if __name__ == "__main__":
    main()







