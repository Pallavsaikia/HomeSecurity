"""
This is starting page
It contains all the link
"""
import time##for timestamp
import datetime##for datetime
import cv2  ##opencv
import os   ##os functions
import dlib ##face detection library
import multiprocessing##for multiprocessing
import psutil## for process utilities
from multiprocessing.managers import BaseManager##base manager to use lifoqueue
from imutils.face_utils import FaceAligner##for aligning face
import numpy##for numpy arrays
from keras.models import load_model##for loading keras model
from keras.preprocessing import image##for preprocessin image for human detection
from queue import LifoQueue##lifoqueue

##package loaded to send data to server
import SendDataToServer.SendServer as ss
##package to be loaded for motion detetion
import motion_sensor.motion_detect as ms
##packages loaded for face detection
import detect_face.face as face
##package loaded for facial land mark detection
import landmark_detection.facial_landmarks as fl


#################################class objects##########################################################
##creating class for motion detection
fgbg=cv2.createBackgroundSubtractorMOG2()
##face detector class object
detector = dlib.get_frontal_face_detector()
##location of facial landmark datafile
path='landmark_detection/shape_predictor_68_face_landmarks.dat'
###facial landmark detector
predictor=dlib.shape_predictor(path)
##face alignment
facealign = FaceAligner(predictor, desiredFaceWidth=500,desiredFaceHeight=500)
##loads model for human detection
classifier=load_model('human_detection/human_recognition.h5')
########################################################################################################

##########################################manager for lifo queue######################################################
class MyManager(BaseManager):
    pass
MyManager.register('LifoQueue', LifoQueue)
#################################################################################################


##########################################video capture object##########################################
#capturing video
#cv2.startWindowThread()##starts threads for windows
#cv2.namedWindow('frame',cv2.WINDOW_NORMAL)##makes the window resizable
#######################################################################################################


##############################main section##############################################################
def main_prog(queue,parallel_program_id,send_data_id,queue_data):##queue and parallel process id
    ################variables###############
    video=cv2.VideoCapture(0)#starting webcam
    child=psutil.Process(parallel_program_id)##psutil for parallel process
    childsendprogram=psutil.Process(send_data_id)##psutil for parallel process
    detect=0##stores 0 if motion is detected or the timestamp
    faceloc=0##locates the face in the frame
    count_in=0##saves cpu usage due to harcassade
    motion_is_sensed=0##binary variable to make efficient switch between motion detection and face detection
    k=0##used to skip number of frames
    t=0##motion not detected print once
    #i=0##for numbering human detection
    switch=0##switching face detection and body detection
    count=0##counts confidence for human
    url = 'http://localhost/paul/human_detection.php'##server location
    #url = 'http://www.pallavsaikia.cu.ma/pallav.php'
    #push_queue=0##counter to select time on when to push to queue
    #i=0
    ########################################

    while True:
            check,frame=video.read()##reads video frame by frame
            send_frame=frame##copying frame
                         
            #################motion detection##################
            if motion_is_sensed ==0 :##this prevents checking for motion again and again
                detect,videoMasked=ms.get_binary_motion_return(frame,fgbg,0)##this function returns 0 if no motion is detected
                """i+=1 
                print(i)
                cv2.imwrite('E:/software/MCA 6TH project/sample code/motion detection/datatemp/nohumansss'+str(i)+'.jpg',videoMasked)
                """
                if k%10==0:
                ##stops checking motion 
                    if detect != 0:##if motion is detected it sets motion_is_sensed to 1 so that it stops looping in itself
                        #print(detect,': motion detected')
                        motion_is_sensed=1
                        t=0
                    else:##if motion is not detected
                        motion_is_sensed=0
                        if str(child.status()) == 'running':
                             
                            child.suspend()##child process suspended
                            while queue.empty() is False:
                                queue.get()
                                
                        if str(childsendprogram.status()) == 'running':
                            childsendprogram.suspend()##child sending data to server process suspended
                            while queue_data.empty() is False:
                                queue_data.get()##data queue is emptied       
                        if t==0:
                            count=0
                            print('\nmotion not detected.camera is sensing for motion')
                            t=1
                ##########################################
                #################face and body############
            if motion_is_sensed ==1:
                    if str(childsendprogram.status()) == 'stopped':
                        childsendprogram.resume()##sending data to server process is resumed if motion is sensed
                    count_in+=1##saves cpu usage due to harcassade
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ##continues looping through next frame if no motion detected
                    #####################face detection#####################
                    if switch ==0:
                        frame,faceloc,status=face.face_locate(frame,gray,detector)##locates face; status=if face is present; faceloc = object of location of face; returns frame with face pointed
                        if status !=0:##if face is detected
                            switch=0
                            motion_is_sensed=1
                            ##################
                            if len(faceloc) > 0:##face is detected
                                #if push_queue % 30 == 0:
                                tup=(faceloc,send_frame)##creates tuple to put it on queue
                                if queue.full() == False:
                                    queue.put(tup)##image stored in multiprocessing queue
                                #push_queue+=1
                                if str(child.status()) == 'stopped':
                                    child.resume()##facial landmark process is resumed
                            else:
                                if str(child.status()) == 'running':
                                    child.suspend()##facial landmark process is suspended
                            ###################
                        else:
                            motion_is_sensed=0
                            switch = 1
                    ################################################
                    
                    ################body detection##################
                    else:##if face is not detected
                        cv2.imwrite('temporary1.jpg',videoMasked)##writes it temporarily 
                        test_image = image.load_img('temporary1.jpg', target_size = (64, 64))##loads the saved image in 64 x 64 size
                        #videoMasked=cv2.dilate(videoMasked.copy(), None, iterations=2)
                        test_image = image.img_to_array(test_image)##converts to numpy array
                        test_image = numpy.expand_dims(test_image, axis = 0)##flattens the image
                        result = classifier.predict(test_image)##predict if human present or not
                        if result[0][0] == 1:
                            #print('no one detected')
                            switch=0
                            motion_is_sensed=0##set to zero is that motion sensing starts again
                            count=0
                        else:
                           
                            count+=1
                            switch=0##set to zero so that it can look for face in second iteration
                            motion_is_sensed=0
                            if count >=3:##confidence that its a human
                                data={'timestamp':time.time(),'time_d':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'des':'human detected','who':'unknown','read_s':0}
                                send_data_toserv=(data,frame,url)
                                if queue_data.full() == False:##if data queue is empty
                                    queue_data.put(send_data_toserv)##puts data on data queue
                                print('human')
                                #i+=1##incremented for name
                                count=0
                
            ################################################
            cv2.imshow('frame',frame)##displaying frames
            cv2.imshow('video masked',videoMasked)##optional to check 
            if cv2.waitKey(1) & 0xFF == ord('q'):   
                break
        ############################################
        
    ###############terminating child process#########################
    parent=psutil.Process(os.getpid())
    for childs in parent.children(recursive=True):  # or parent.children() for recursive=False
        childs.kill()
    parent.kill()
    ##################################################################
        
    cv2.destroyAllWindows()
    video.release()         
#######################################################################################################

    

##################################################main function########################################
if __name__=='__main__':
    manager = MyManager()
    manager.start()
    queue = manager.LifoQueue()##lifo queue
    queue_data=manager.LifoQueue()
    child = multiprocessing.Process(target=fl.get_landmarks,args=(queue,predictor,facealign,detector,queue_data))##child process
    child.daemon=True##makes the process run in background
    child.start()
    child_send = multiprocessing.Process(target=ss.senddata,args=(queue_data,))##child sending data process
    child_send.daemon=True##makes the process run in background
    child_send.start()
    main_prog(queue,child.pid,child_send.pid,queue_data)
#####################################################################################################