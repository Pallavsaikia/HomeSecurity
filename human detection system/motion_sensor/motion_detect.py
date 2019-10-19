"""
This file is used to detect motion
Importing opencv and time package

get_binary_motion_return(frame,fgbg,status)
"""
#####################################import libraries#########################################
import cv2,time
#############################################################################################

############################################body###################################################
def get_binary_motion_return(frame,fgbg,status):##this function returns if motion is detected or not
    ##frame is the input frame
    ##fgbg is an object of cv2.createBackgroundSubtractorMOG2()
    ##status is binary value;0 means motion not detected ;1 otherwise
    videoMasked=fgbg.apply(frame)##applies createBackgroundSubtractorMOG2() on the inputted frame
    ##returns number of contour detected
    (_,cnts,_)=cv2.findContours(videoMasked.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:##10000 is the threshold value
            continue   ##continues to other contour if hreshold lower than 10000
        status=1 ##set status to 1 if contour detected
        ##(x, y, w, h)=cv2.boundingRect(contour)##optional to check
        ##cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)##optional to check
    
    ##cv2.imshow('frame',frame)##optional to check
    if status == 1:##if motion is detected time is returned
        return (time.time(),videoMasked)
    else:
        return (0,videoMasked)
###############################################################################################