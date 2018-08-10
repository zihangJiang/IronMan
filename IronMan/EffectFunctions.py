#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from skimage import img_as_float
import numpy as np
import numpy.matlib
import math
import face_recognition
import cv2

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("we.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [obama_face_encoding]
known_face_names = ["Barack Obama"]


Explode_gif = cv2.VideoCapture('explode.gif')
Laser_gif = cv2.VideoCapture('effect.gif')
for i in range(3):
    Laser_gif.read()
just = [1,1,1]



def Effect(img):
    img = img_as_float(img)

    row, col, channel = img.shape
    img_out = img * 1.0
    alpha = 70.0
    beta = 30.0
    degree = 20.0
    
    center_x = (col-1)/2.0
    center_y = (row-1)/2.0
    
    xx = np.arange(col)
    yy = np.arange(row)
    
    x_mask = numpy.matlib.repmat (xx, row, 1)
    y_mask = numpy.matlib.repmat (yy, col, 1)
    y_mask = np.transpose(y_mask)
    
    xx_dif = x_mask - center_x
    yy_dif = center_y - y_mask
    
    x = degree * np.sin(2 * math.pi * yy_dif / alpha) + xx_dif
    y = degree * np.cos(2 * math.pi * xx_dif / beta) + yy_dif
    
    x_new = x + center_x
    y_new = center_y - y 
    
    int_x = np.floor (x_new)
    int_x = int_x.astype(int)
    int_y = np.floor (y_new)
    int_y = int_y.astype(int)
    
    for ii in range(row):
        for jj in range (col):
            new_xx = int_x [ii, jj]
            new_yy = int_y [ii, jj]
    
            if x_new [ii, jj] < 0 or x_new [ii, jj] > col -1 :
                continue
            if y_new [ii, jj] < 0 or y_new [ii, jj] > row -1 :
                continue
    
            img_out[ii, jj, :] = img[new_yy, new_xx, :]
    return img_out




def AddRect(frame):

    small_frame = frame
    rgb_small_frame = small_frame[:, :, ::-1]


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#        top *= 4
#        right *= 4
#        bottom *= 4
#        left *= 4
#        
        # Draw a box around the face
        #temp = frame[top:bottom,left:right,:]
        #frame[top:bottom,left:right,:] = Effect(temp)
        
        #temp = frame[top:bottom,left:right,:]
        #temp = Effect(temp)*255
        #frame[top:bottom,left:right,:] = temp
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return (True,frame)


def AddTarget(frame):
    aim_img = cv2.imread('target.jpg')
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
        name = "Target"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        effe = cv2.resize(aim_img, (right - left,bottom - top))
        #effe = effe[:h-2,:w-2,:]
        
        if left<0 or top<0 or right>frame.shape[0] or bottom>frame.shape[1]:
            continue
        
        ph,pw = (max(left,0),max(top,0))
        phr,pwr = (min(right,frame.shape[0]),min(bottom,frame.shape[1]))
        
        effe = effe[:pwr-pw,:phr-ph,:]
        mask = effe.copy()[:,:,:1]

        thresh = 5
        mask[mask<=thresh] = 0
        mask[mask>thresh] = 1

        
        
        
        frame[pw:pwr,ph:phr,:] = frame[pw:pwr,ph:phr,:] * (1-mask)
        frame[pw:pwr,ph:phr,:] = frame[pw:pwr,ph:phr,:]  + effe*mask
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return (True,frame)


def AddExplode(frame):
    for i in range(5):
        success, current_gif = Explode_gif.read()
    if not success:

        return False,frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]


    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
        name = "Target"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        rate = 2
        
        top *= rate
        right *= rate
        bottom *= rate
        left *= rate
        
        bigger = 40
        top -= bigger
        right += bigger
        bottom += bigger
        left -= bigger
        
        if left<0 or top<0 or right>frame.shape[1] or bottom>frame.shape[0]:
            continue
        
            
        effe = cv2.resize(current_gif, (right - left, bottom - top))

        #effe = effe[:h-2,:w-2,:]
        #h,w,_ = effe.shape
        #ph,pw = (max(left,0),max(top,0))
        #phr,pwr = (min(right,frame.shape[0]),min(bottom,frame.shape[1]))
        ph,pw = (left,top)
        phr,pwr = (right,bottom)
        #effe = cv2.resize(current_gif, (right - left,bottom - top))
        mask = effe.copy()[:,:,:1]

        thresh = 5
        mask[mask<=thresh] = 0
        mask[mask>thresh] = 1

        
        
        
        frame[pw:pwr,ph:phr,:] = frame[pw:pwr,ph:phr,:] * (1-mask)
        frame[pw:pwr,ph:phr,:] = frame[pw:pwr,ph:phr,:]  + effe*mask




        
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, 'Exploding', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return (True,frame)

def rot(frame1,frame2,p1,p2,pp1,pp2):

    rows1,cols1,a = frame1.shape
    rows2,cols2,a = frame2.shape
    rows=max(rows1,rows2)
    cols=max(cols1,cols2)
    whole = np.zeros_like(frame1)
    whole = cv2.resize(whole,(cols,rows))
    whole[:rows1,:cols1,:]=frame1
    p3=[p2[1]-p1[1]+p1[0],-p2[0]+p1[0]+p1[1]]
    pp3=[pp2[1]-pp1[1]+pp1[0],-pp2[0]+pp1[0]+pp1[1]]
    pts1 =np.float32([p1,p2,p3])
    pts2 =np.float32([pp1,pp2,pp3])
    rot_mat = cv2.getAffineTransform(pts1,pts2)
    res = cv2.warpAffine(whole,rot_mat,(cols2,rows2)) 
    return res



def AddLaser(frame):
    if just[0] == 1:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]
    
    
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.4)
            name = "Target"
    
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
    
            face_names.append(name)
        just[0] = 0
        just[1] = face_locations
        just[2] = face_names
    else:
        face_locations = just[1]
        face_names = just[2]
            
    for i in range(2):
        
        success, current_gif = Laser_gif.read()
    if not success:
        just[0] = 1
        return False,frame
    current_gif = current_gif[5:current_gif.shape[0]-2,:current_gif.shape[1]-12,:]


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        rate = 2
        
        top *= rate
        right *= rate
        bottom *= rate
        left *= rate
        

            
        #effe = cv2.resize(current_gif, (right - left, bottom - top))
        effe = rot(current_gif, frame, [int(current_gif.shape[0]/2), 0],[int(current_gif.shape[0]/2), current_gif.shape[1]]\
            ,[int((left+right)/2),top-20],[int(frame.shape[0]/2),frame.shape[1]])
        #effe = effe[:h-2,:w-2,:]
        #h,w,_ = effe.shape
        #ph,pw = (max(left,0),max(top,0))
        #phr,pwr = (min(right,frame.shape[0]),min(bottom,frame.shape[1]))
        ph,pw = (left,top)
        phr,pwr = (right,bottom)
        #effe = cv2.resize(current_gif, (right - left,bottom - top))
        mask = effe.copy()[:,:,:1]

        thresh = 210
        mask[mask<=thresh] = 0
        mask[mask>thresh] = 1
        
        frame = frame * (1-mask)
        frame = frame  + effe*mask
        
        
        
#        frame[pw:pwr,ph:phr,:] = frame[pw:pwr,ph:phr,:] * (1-mask)
#        frame[pw:pwr,ph:phr,:] = frame[pw:pwr,ph:phr,:]  + effe*mask

    return (True,frame)



