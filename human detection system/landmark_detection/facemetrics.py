# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 18:56:39 2018

@author: paul
"""

import math

def face_metric(landmarks):
    metric=[]
    ##########################################distance between upper and lower lips##################
    a= math.sqrt(((landmarks[62][0]-landmarks[66][0])**2)+((landmarks[62][1]-landmarks[66][1])**2))
    metric.append(a)
    ##################
    
    ##########################################distance between corner of lips##################
    a= math.sqrt(((landmarks[54][0]-landmarks[48][0])**2)+((landmarks[54][1]-landmarks[48][1])**2))
    metric.append(a)
    #####################################
    
    ################################################angle between corner of lips###########################
    vA = [(landmarks[48][0]-landmarks[57][0]), (landmarks[48][1]-landmarks[57][1])]
    vB = [(landmarks[57][0]-landmarks[54][0]), (landmarks[57][1]-landmarks[54][1])]
    metric.append(findangle(vA,vB))
    ############################################3
    
    ##################################distance between eyebrows##############################3
    a= math.sqrt(((landmarks[21][0]-landmarks[22][0])**2)+((landmarks[21][1]-landmarks[22][1])**2))
    metric.append(a)
    ################################################
    
    ######################################distance between inner eyecorner and inner eyebrows-left############
    a= math.sqrt(((landmarks[21][0]-landmarks[39][0])**2)+((landmarks[21][1]-landmarks[39][1])**2))
    metric.append(a)
    ######################################################
    
    ######################################distance between inner eyecorner and inner eyebrows-right############
    a= math.sqrt(((landmarks[22][0]-landmarks[43][0])**2)+((landmarks[22][1]-landmarks[43][1])**2))
    metric.append(a)
    ######################################################
    
    
    
    
    
    return metric




def findangle(vA,vB):
    dot_prod = dot(vA, vB)
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg

    
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]  
    