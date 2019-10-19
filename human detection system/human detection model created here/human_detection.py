# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:01:15 2018

@author: paul
"""

"""
This file is used to detect motion
"""
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image
classifier=load_model('human_recognition.h5')
video=cv2.VideoCapture(0)
i=0
count=0
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#connectivity = 4
fgbg=cv2.createBackgroundSubtractorMOG2()
while True:
    
    ret ,frame =video.read()
    videoMasked=fgbg.apply(frame)
    #videoMasked= cv2.morphologyEx(videoMasked, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('lol.jpg',videoMasked)
    test_image = image.load_img('lol.jpg', target_size = (64, 64))
    
    #videoMasked=cv2.dilate(videoMasked.copy(), None, iterations=2)
    cv2.imshow('video mased',videoMasked)
    cv2.imshow('image',frame)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        print('not human')
        count=0
    else:
        count+=1
        if count >=20:
            print('human')
            count=0
        
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()