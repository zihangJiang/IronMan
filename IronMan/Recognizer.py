#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import face_recognition


def Rotation(x1, x2, center, theta):
    from math import atan, sin, cos
    d1 = x1 - center
    d2 = x2 - center
    if d2==0:
        return theta
    if d1==0:
        return 0
    alpha = d1/d2
    theta1 = 2*atan(sin(theta/2)/(cos(theta/2) - 1/alpha))
    return theta1

class Recognizer():
    def __init__(self, img_path = "we.jpg"):
        # Load a sample picture and learn how to recognize it.
        obama_image = face_recognition.load_image_file(img_path)
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
        
        # Create arrays of known face encodings and their names
        self.known_face_encodings = [obama_face_encoding]
        self.known_face_names = ["Barack Obama"]
        
    def FindTargetFace(self, frame):
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if len(face_encodings)==0:
            return False,0,0,0,0
        
        #face_names = ["Barack Obama"]
        matches=[]
        location_index = 0
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding,tolerance=0.4)
            if True in matches:
                #first_match_index = matches.index(True)
                location_index = face_encodings.index(face_encoding)
                break
            
        top, right, bottom, left = face_locations[location_index]
        return True, top, right, bottom, left
    
    def SolveRotation(self, frame1, frame2, angle1, angle2, Horizental = True):
        valid, top, right, bottom, left = self.FindTargetFace(frame1)
        valid2, top2, right2, bottom2, left2 = self.FindTargetFace(frame2)
        # if people are not in both photo
        if not (valid and valid2):
            if valid:
                print('people only in frame_start')
            if valid2:
                print('people only in frame_end')
            print('please choose a new angle')
            return -1
        center = frame1.shape
        delta_angle = angle2 - angle1
        lr_rot = Rotation((right+left)/2,(right2+left2)/2,center[1]/2,delta_angle/180*3.14)/3.14*180
        tb_rot = Rotation((top+bottom)/2,(top2+bottom2)/2,center[0]/2,delta_angle/180*3.14)/3.14*180
        if Horizental:
            rot=lr_rot
        else:
            rot=tb_rot
        print(rot)
        return rot