from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imutils import paths
from scipy import io
import numpy as np
import random
import imutils
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/pi/faces/lbpFace.yml')
cascadePath = "lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

camera = cv2.VideoCapture(0)

def getCameraRead():
    (grabbed, frame) = camera.read()
    return cv2.flip(frame, 1)

while True:
    frame = getCameraRead()
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

camera.release()
cv2.destroyAllWindows()
