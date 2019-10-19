"""
this file sends data to server
"""

#############################################libraries######################################
import requests#SENDINNG post request to server
import cv2 
import time
##############################################################################################

##########################################main function#######################################################
def senddata(queue_data):
    while True:##runs infinitely
        while queue_data.empty() is False:##if queue isnt empty
            (data,frame,url)=queue_data.get()##get data from queue
            cv2.imwrite('temporary_colored.jpg',frame)
            files = {'file1': open('temporary_colored.jpg', 'rb')}
            try:
                requests.post(url,data=data,files=files)
            except:
                raise
            time.sleep(0.3)
        while queue_data.empty() is True:##while the queue is empty
            continue
############################################################################################