import glob
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from scipy import io
import numpy as np
import imutils
import cv2
import signal
import sys
import time
import datetime

bossName = "boss"
cam_id = 0
monitor_winSize = (640, 480)
cam_resolution = (1024,768)

#face_cascade = cv2.CascadeClassifier('objects\\haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('objects\\lbpcascade_frontalface.xml')

last_detect = 0
face_size = (47, 62)
target = np.loadtxt('objects\\target.out', delimiter=',', dtype=np.str)
le = LabelEncoder()
le.fit_transform(target)


print("The model is loading , please wait...")
#recognizer = cv2.face.createLBPHFaceRecognizer(radius=2, neighbors=16, grid_x=8, grid_y=8)
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
recognizer.read("objects\\boss.yaml")
	
camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])

def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
    return image

def signal_handler(signal, frame):
    print('Stop boss detection')
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while True:
    (grabbed, img) = camera.read()    

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= 1.1,
        minNeighbors=8,
        minSize=face_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face = cv2.resize(roi_gray, face_size)
        
        (prediction, conf) = recognizer.predict(face)
        namePredict = le.inverse_transform(prediction)
        print(namePredict, conf)
        #putText(img, namePredict+"["+str(int(conf))+"]", x, y, (255,35,35), thickness=5, size=1)
        putText(img, namePredict, x, y, (255,35,35), thickness=4, size=2)  
		
        if(bossName in namePredict):
            last_detect = time.time()
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
            cv2.namedWindow("Desktop", cv2.WINDOW_NORMAL)        # Create a named window
            cv2.setWindowProperty("Desktop", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
            cv2.moveWindow("Desktop", 0,0)
            cv2.imshow("Desktop", cv2.imread("objects\\desktop.jpg"))
            cv2.waitKey(1)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    if(last_detect>0):
        last_boss_time = datetime.datetime.fromtimestamp( int(last_detect) ).strftime('%Y-%m-%d %H:%M:%S')
        putText(img, last_boss_time + " boss is here!", 5, 40, (255,35,35), thickness=2, size=1)  
    			
    r = monitor_winSize[1] / img.shape[1]
    dim = (monitor_winSize[0], int(img.shape[0] * r))
    img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Frame", img2)
    key = cv2.waitKey(1)
    
