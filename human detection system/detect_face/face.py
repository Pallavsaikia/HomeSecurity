"""
This file is used to detect faces and return face location and status of the face detection process

face_locate(frame,gray,detector)
"""
##############################import libraries############################################
import cv2 ## opencv
##########################################################################################


############################################variables#####################################
face_detect_stats=0##status if face is detected
##########################################################################################


########################################helper functions###################################
## Fancy box drawing function by Dan Masek
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2

    ## Top left drawing
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    ## Top right drawing
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    ## Bottom left drawing
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    ## Bottom right drawing
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
#########################################################################################


###################################main function########################################    
def face_locate(frame,gray,detector):
        
        face_detect_stats=0#status if face is detected 0,if no detected
        
        ## Make copies of the frame for transparency processing
        overlay = frame.copy()
        output = frame.copy()

        ## set transparency value
        alpha  = 0.5

        ## detect faces in the gray scale frame
        face_rects = detector(gray, 0)

        ## loop over the face detections
        for i, d in enumerate(face_rects):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            #print('face located at ',d)##face locatio-top left
            face_detect_stats=1##set to 1 if face is detected
            ## draw a fancy border around the faces
            draw_border(overlay, (x1, y1), (x2, y2), (162, 255, 0), 2, 10, 10)

        ## make semi-transparent bounding box
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        ## show the frame
        return (output,face_rects,face_detect_stats)
#############################################################################################