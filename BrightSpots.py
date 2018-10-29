##  This program will outlay facial landmarks ( 68 pt or 5 pt ) on the face captured. 
## Aditionally it will remove any bright white spots ( Total Pixel Count > 8000 ) from the image.
## Usage python BrightSpots.py 68 ( or 5)

import cv2
import dlib
import scipy.misc
import numpy as np
import os
from skimage import measure
from imutils import face_utils
import argparse
import imutils
import sys

# DLIBs Frontal Face Detector
face_detector = dlib.get_frontal_face_detector()
landmarks = sys.argv[1]
# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face

if(landmarks == '68'):
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
else:
    shape_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

match="Harshul"
frame_count=-1
import time
if(match != 'Not Found' and match!=""):
    cam2 = cv2.VideoCapture(0)
    time.sleep(2.0)
    cv2.namedWindow("Output_Face_Encodings")
    while True:
        ret, frame = cam2.read()
        #cv2.imshow("Output_Face_Encodings", frame)
        #win.clear_overlay()
        frame_count+=1
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        thresh = cv2.threshold(blurred, 230, 255, 0)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        for label in np.unique(labels):
           if label == 0:
               continue 
           labelMask = np.zeros(thresh.shape, dtype="uint8")
           labelMask[labels == label] = 255
           numPixels = cv2.countNonZero(labelMask)
           if numPixels > 8000:                                 # remove only if pixel count > 8000
              mask = cv2.add(mask, labelMask)
        cv2.bitwise_not(frame,frame,mask)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        image_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        Y, Cr, Cb = cv2.split(image_YCrCb)
        Y = cv2.equalizeHist(Y)
        image_YCrCb = cv2.merge([Y, Cr, Cb])
        frame = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCR_CB2BGR)
        invGamma = 1.0 / 1.0
        table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
        cv2.LUT(frame, table)
        gray = cv2.resize(frame,None,fx=0.25, fy=0.25,interpolation=cv2.INTER_AREA) 
        if not ret:
            break
        if(frame_count%3==0):
            detected_faces = face_detector(gray, 1)
        #print(frame_count)
        for (i, rect) in enumerate(detected_faces):
            shape = shape_predictor(gray, rect)
            if(landmarks == '68'):
                shape = face_utils.shape_to_np(shape)
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                #cv2.rectangle(frame, (x*4, y*4), ((x + w)*4, (y + h)*4), (0, 255, 0), 2)
                #cv2.putText(frame, "Face #{}".format(match), (x*4 - 10, y*4 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for (x, y) in shape:
                    cv2.circle(frame, (x*4, y*4), 2, (0, 0, 255), -1)
            else:
                vec = np.empty([5, 2], dtype = int)
                for b in range(5):
                    vec[b][0] = shape.part(b).x
                    vec[b][1] = shape.part(b).y
                cv2.line(frame,(4*vec[0][0],4*vec[0][1]),(4*vec[1][0],4*vec[1][1]),(0, 0, 255),2)
                cv2.line(frame,(4*vec[3][0],4*vec[3][1]),(4*vec[2][0],4*vec[2][1]),(0, 0, 255),2)
                cv2.line(frame,(4*vec[1][0],4*vec[1][1]),(4*vec[4][0],4*vec[4][1]),(0, 0, 255),2)
                cv2.line(frame,(4*vec[3][0],4*vec[3][1]),(4*vec[4][0],4*vec[4][1]),(0, 0, 255),2)
        cv2.imshow("Output_Face_Encodings",frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
    cam2.release()
    cv2.destroyAllWindows()












