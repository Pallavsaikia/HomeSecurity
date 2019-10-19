"""
this file extracts facial landmarks and stores it in a csv file

"""
###############################################################################
import dlib
import cv2
import numpy
import os
import detect_face.face as face
import csv

import landmark_detection.facemetrics as fm

################################################################################

################################################################################
##declaration here
person=input('enter the name:')
path_read_from='database/preprocessed/aligned/'+person+'/'
path_save_to='database/preprocessed/full preprocessed/'+person+'/'
path_save_all_csv_to='database/preprocessed/full preprocessed/all.csv'
path='landmark_detection/shape_predictor_68_face_landmarks.dat'
listing=os.listdir(path_read_from)
detector = dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(path)

#################################################################################

##################################################################################
def extract_face():
    
    j=0
    for file in listing:#file stores filename with extension
        frame=cv2.imread(path_read_from + file)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _,faceloc,_=face.face_locate(frame,gray,detector)
        for i,d in enumerate(faceloc):
            landmarks=numpy.matrix([[p.x,p.y] for p in predictor(gray,dlib.rectangle(d.left(),d.top(),d.right(),d.bottom())).parts()])    
            lm = numpy.squeeze(numpy.asarray(landmarks))
            frame=annotate_landmarks (frame,landmarks)
            cv2.imwrite(path_save_to+file,frame)
            face_metrics=fm.face_metric(lm)
            face_metrics.append(person)
            with open(path_save_to+person+'.csv','a', newline='') as f_handle:##save a file
                    thewriter=csv.writer(f_handle,delimiter=',')
                    thewriter.writerow(face_metrics)
            with open(path_save_all_csv_to,'a', newline='') as f_handle:##save a file
                    thewriter1=csv.writer(f_handle,delimiter=',')
                    thewriter1.writerow(face_metrics)
            """frame=annotate_landmarks (frame,landmarks)
            cv2.imwrite(path_save_to+file,frame)
            
            landmark_transpose=numpy.transpose(landmarks)##transpose (2,68) matrix to (68,2) matrix
            subtracting_matrix = [[d.left()-10],[d.top()-10]]##matrix for subtracting and normalise
            normalised_matrix=landmark_transpose-subtracting_matrix##normalise the matrix
            normalised_matrix=normalised_matrix.ravel()##flaten the martix to 136
            FaceArray = numpy.squeeze(numpy.asarray(normalised_matrix))##squeezes the matrix
            save_in_csv=[p for p in FaceArray]##spreading the elements of the matrix 
            save_in_csv.append(person)##append the name
            
            with open(path_save_to+person+'.csv','a', newline='') as f_handle:##save a file
                    thewriter=csv.writer(f_handle,delimiter=',')
                    thewriter.writerow(save_in_csv)
            with open(path_save_all_csv_to,'a', newline='') as f_handle:##save a file
                    thewriter1=csv.writer(f_handle,delimiter=',')
                    thewriter1.writerow(save_in_csv)"""
        j+=1
        print(j)
        #get_landmarks(faceloc,frame)
        """filename, file_extension = os.path.splitext(file)#serparating filename, file_extension 
        im.save(path_save_to+filename+'.jpg')#saving as jpg file in different directory"""
        
    print('done')
##################################################################################
    
##################################################################################
def annotate_landmarks (im,landmarks):##displays landmarks
    font = cv2.FONT_HERSHEY_SIMPLEX
    im.copy()##copieis image
    for idx,point in enumerate(landmarks):##iterate through the landmark
        pos=(point[0,0],point[0,1])##centre of the landmarks
        cv2.putText(im, str(idx), pos, font, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
    return im
##################################################################################
extract_face()
    