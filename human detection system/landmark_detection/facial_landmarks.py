"""
this file is used to extract facial features.it runs in parallel using multi processing

get_landmarks(queue,predictor,facealign,detector);queue is the queue,predictor is facecial landmark predictor,fae_align to align faces,detector for detecting faces
"""
#######################################import libraries################################################
import time##for timestamp
import datetime##for datetime
import numpy
import dlib
import cv2
import pandas as pd
import detect_face.face as face
from sklearn.preprocessing import LabelBinarizer ##for encoding
from keras.models import load_model##load model

import landmark_detection.facemetrics as fm
#############################################variables#########################################################
#sharpening=numpy.array([[-1,-1,-1],
#                        [-1,9,-1],
#                        [-1,-1,-1]])
#alpha=1.4
#beta=0
##face model to predict whose faces are detected
face_model=load_model('E:/software/MCA 6TH project/stranger detection system/model/face_recognition.h5')
##names of people to be detected
face_names=pd.read_csv('E:/software/MCA 6TH project/stranger detection system/model/savednames.csv')
url = 'http://localhost/paul/human_detection.php'##to store the picture
Y=face_names.iloc[:,0:1].values##names of faces
encoder = LabelBinarizer()##transform to localbinarizer
y = encoder.fit_transform(Y)
url = 'http://localhost/paul/human_detection.php'
#url = 'http://www.pallavsaikia.cu.ma/pallav.php'
#######################################################################################################

#########################################main function#################################################
def get_landmarks(queue,predictor,facealign,detector,queue_data):
    i=0
    while True:#runs forever
        while queue.empty() is False: ##while the queueueue is full
                tup = queue.get()##gets the tuple from the queue
                for i in range(3):
                    queue.get()
                (faceloc,frame)=tup##frame and face location extracted
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                #edited= cv2.addWeighted(frame,alpha,numpy.zeros(frame.shape,frame.dtype),0,beta)
                #edited = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #edited=cv2.filter2D(frame,-1,sharpening)
                #dst = gamma_correction(frame, 0.5)
                cv2.imwrite('temporary_context.jpg',frame)
                time_now=time.time()
                date_time_now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)##converts to grayscale
                l=list() 
                l.clear()
                for j, d in enumerate(faceloc):##loops through faces in the frame
                    ####################################align face############################################
                    detected_rect=dlib.rectangle(d.left()-30, d.top()-30, d.right()+30, d.bottom()+30)##rectangle with the location of the faces
                    faceAligned = facealign.align(frame, gray, detected_rect)##aligns the face
                    ###########################################################################################
                    
                    #############################################get face matrix##################################
                    grayed_aligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)##converts the aligned faces
                    not_needed1,aligned_face_loc,not_needed2=face.face_locate(faceAligned,grayed_aligned,detector)##gets the location of the aligned face
                    for m, n in enumerate(aligned_face_loc):##loops through aligned faces
                        landmarks=numpy.matrix([[p.x,p.y] for p in predictor(grayed_aligned,dlib.rectangle(n.left(),n.top(),n.right(),n.bottom())).parts()])##returns face landmark numpy array
                        ##displays location of face landmarks
                    #############################################################################################    
                        lm = numpy.squeeze(numpy.asarray(landmarks))
                        face_metrics=fm.face_metric(lm)
                        face_metric=numpy.asarray(face_metrics)
                        a = numpy.asmatrix(face_metric)
                    ##########################normalise face matrix###############################
                        detected_face=predict_face(a,face_model)##name of the face detected
                        
                        """files = {'file1': open('temporary_face.jpg', 'rb'),'file2': open('temporary_context.jpg', 'rb')}
                        data={'timestamp':time_now,'time_d':date_time_now,'des':'face detected','who':detected_face,'read_s':0}
                        requests.post(url,data=data,files=files)"""
                        if len(l) <=0:##creates the list
                            l=[detected_face]
                        else:##appends in the list
                            l.append(detected_face)
                if l=='':
                    l='unknown'
                k=",".join(str(x) for x in l)
                data={'timestamp':time_now,'time_d':date_time_now,'des':'face detected','who':k,'read_s':0}
                send_data_toserv=(data,frame,url)
                if queue_data.full() == False:
                    queue_data.put(send_data_toserv)##puts data on server send queue
                        
                    #########################################################################
                print(l)
                time.sleep(0.2) ##sleeps for 0.5 second to lower cpu usage
                i+=1
        while queue.empty() is True:##while the queue is empty
            continue
####################################################################################################################

        
################################################gamma correction#########################################################
def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return numpy.uint8(img*255)
###################################################################################################################
    
############################################predict face landmarks###########################################
def predict_face(face_metrics,face_model):

    prediction=face_model.predict(face_metrics)##predict face
    prediction=(prediction>0.6)##confidence
    if prediction.any() == True:
        return (encoder.inverse_transform(prediction)[0])
    else:
        return ('unknown')
################################################################################################################
