"""
This file is used extract faces for model training
"""
############################################################################
import cv2 
import detect_face.face as face
import dlib
import os
############################################################################

#############################################################################
detector = dlib.get_frontal_face_detector()
############################################################################

#############################################################################
bottomLeftCornerOfText=(10,80)
fontScale=1
fontColor=(0,255,0)
lineType= 2
font= cv2.FONT_HERSHEY_SIMPLEX
################################################################################

#################################################################################
def getface():
    person=input('enter the name:')
    os.makedirs('database/raw/'+person, exist_ok=True)
    os.makedirs('database/preprocessed/aligned/'+person, exist_ok=True)
    os.makedirs('database/preprocessed/full preprocessed/'+person, exist_ok=True)
    video=cv2.VideoCapture(0)
    num_times=0
    while True:
        ret,frame=video.read()##reads frame from webcam video
        frame1,faceloc,status=face.face_locate(frame,frame,detector)##locates face; status=if face is present; faceloc = object of location of face; returns frame with face pointed
        if status == 1:##if face is found
            ########################################3
            cv2.imwrite('database/raw/'+person+'/'+person+str(num_times)+'.jpg',frame)
            num_times+=1
            ############################################
        if num_times >=1000:##if enough pictures are taken
            cv2.putText(frame1,"YOU CAN STOP",(10,250),font,1,fontColor,lineType)
        cv2.putText(frame1,str(num_times),bottomLeftCornerOfText,font,1,fontColor,lineType)
        cv2.putText(frame1,"turn your head slowly in all direction",(10,50),font,0.75,fontColor,lineType)    
        cv2.imshow('frame',frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):   
                break
    cv2.destroyAllWindows()
    video.release()
###################################################################################################
getface()