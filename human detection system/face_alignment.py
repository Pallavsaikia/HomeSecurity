"""
this file is used to align faces
"""
#########################################################################libraries################################################################
import dlib
from imutils.face_utils import FaceAligner##for aligning faces
import cv2
import os
#################################################################################################################################################


#####################################################################input name#################################################################
person=input('enter the name:')
###############################################################################################################################################


#########################################################################variables#############################################################
detector = dlib.get_frontal_face_detector()##face detector
predictor = dlib.shape_predictor('landmark_detection/shape_predictor_68_face_landmarks.dat')##landmark predictor
path_read_from='E:/software/MCA 6TH project/stranger detection system/database/raw/'+person+'/'
listing=os.listdir(path_read_from)
path_save_to='E:/software/MCA 6TH project/stranger detection system/database/preprocessed/aligned/'+person+'/'
face_align = FaceAligner(predictor, desiredFaceWidth=500,desiredFaceHeight=500)
################################################################################################################################################


#########################################################aligning face module################################################################

def align():
    for file in listing:#file stores filename with extension
            frame=cv2.imread(path_read_from + file)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for i,d in enumerate(rects):
                face_rect=dlib.rectangle(d.left()-30, d.top()-30, d.right()+30, d.bottom()+30)
                faceAligned = face_align.align(frame, gray, face_rect)##aligned face
                cv2.imwrite(path_save_to+file,faceAligned)##save face
            print(file)
    print('done')
################################################################################################################################################
align()