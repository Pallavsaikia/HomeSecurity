"""
This file is used to detect motion
"""
import numpy as np
import cv2

video=cv2.VideoCapture(0)
i=0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
fgbg=cv2.createBackgroundSubtractorMOG2()
ret ,frame =video.read()
while i<=1200:
    i+=1
    
    ret ,frame =video.read()
    
    videoMasked1=fgbg.apply(frame)
    if i<=5:
        continue
    #videoMasked= cv2.morphologyEx(videoMasked1, cv2.MORPH_OPEN, kernel)
    #videoMasked_dilate=cv2.dilate(videoMasked.copy(), None, iterations=2)
    cv2.imwrite('datatemp/humanssss'+str(i)+'.jpg',videoMasked1)
    cv2.imshow('video mased1',videoMasked1)
    #cv2.imshow('video mased',videoMasked)
    #cv2.imshow('videoMasked_dilate',videoMasked_dilate)
    print(i)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()