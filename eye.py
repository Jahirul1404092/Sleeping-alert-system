# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 02:25:08 2021

@author: Amsa
"""

import playsound
import os
# Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt
cap=cv2.VideoCapture(0)
eyes_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
count=0
while(1):
    ret,img=cap.read()
    facefound=False
    
    # this is for gray,invert,dehaze,invert
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    invert_image=cv2.bitwise_not(gray)
    haze_reduced_image = cv2.fastNlMeansDenoising(invert_image,None,10,7,21)
    invert_again_image=cv2.bitwise_not(haze_reduced_image)
    #to this
    
    faces_rects = face_cas.detectMultiScale(invert_again_image, scaleFactor = 1.2, minNeighbors = 5)
    for (x,y,w,h) in faces_rects:
        facefound=True
        #x,y,w,h=faces_rects
        #for (x,y,w,h) in faces_rects:
            #cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face=invert_again_image[y:y+h, x:x+w]
        #grayface=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        invert_imageface=cv2.bitwise_not(face)
        haze_reduced_imageface = cv2.fastNlMeansDenoising(invert_imageface,None,10,7,21)
        invert_again_imageface=cv2.bitwise_not(haze_reduced_imageface)
        #face=img[y:y+h, x:x+w]
        #gray=cv2.cvtColor(cv2.bitwise_not(cv2.fastNlMeansDenoisingColored(cv2.bitwise_not(face),None,10,10,7,21)),cv2.COLOR_BGR2GRAY)
        
        #gray=cv2.cvtColor(cv2.bitwise_not(cv2.fastNlMeansDenoisingColored(cv2.bitwise_not(img),None,10,10,7,21)),cv2.COLOR_BGR2GRAY)
        #img=cv2.imread('IMG_097639.jpg')
        #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #plt.imshow(test_image_gray, cmap='gray')
        eyes = eyes_cas.detectMultiScale(face, scaleFactor = 1.2, minNeighbors = 3)
        print('eyes found: ', len(eyes))
        #os.system("mpg321 ")
        if(len(eyes)==0):
            count+=1
        if(count==1):
            count=0
            playsound.playsound("music.mp3", True)#ghumocche
        for (x,y,w,h) in eyes:
            cv2.rectangle(face, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if(facefound==False):
        print("Low light")
    plt.imshow(convertToRGB(face))
    
    k=cv2.waitKey(30)
    if k==27:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()