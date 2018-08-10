#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 19:56:33 2018

@author: jzh
"""

from multiprocessing import Process, Queue
import time
from Camera import Camera
from CloudTable import CloudTable
from Recognizer import Recognizer
import cv2

def ShowVedio(q_in, q_out, src=0):
    cam = Camera(src)
    cam.ShowVedio(q_in, q_out)
    
def ControlCT(q_CT):
    ct = CloudTable('/dev/ttyACM0')
    while True:
        if not q_CT.empty():
            signal = q_CT.get(True)
            if signal!=None:
                if signal[0]=='h':
                    ct.set_angle(lr_angle=int(signal[1:]))
                if signal[0]=='v':
                    ct.set_angle(tb_angle=int(signal[1:]))
                if signal[0]=='l':
                    ct.set_light(signal)
                
                if signal=='over':
                    break
                
    del ct

def GetPict(q_in, q_out):
    q_in.put('laizhangtu')
    img = q_out.get()
    return img




def RotateCT(q_CT,signal,sleep_time = 2):
    q_CT.put(signal)
    time.sleep(sleep_time)


def Light(signal):
    q_CT.put(signal)



q_in = Queue()
q_out = Queue()
q_CT = Queue()

#reco = Recognizer()


def LocateFace(v_angle, h_angle, delta_angle = 15):
    RotateCT(q_CT, 'h'+str(h_angle),sleep_time=0.1)
    RotateCT(q_CT, 'v'+str(v_angle))
    frame = GetPict(q_in, q_out)
    cv2.imwrite('frame.jpg',frame)
    valid, top, right, bottom, left = reco.FindTargetFace(frame)
    if valid:
        frame_start = frame
        h_start_angle = h_angle
        v_start_angle = v_angle

        print('find in frame h{},v{}'.format(h_angle,v_angle))
        if left+right<frame.shape[1]:
            h_delta_angle = -delta_angle
        else:
            h_delta_angle = delta_angle
            
        if top + bottom < frame.shape[0]:
            v_delta_angle = -delta_angle
        else:
            v_delta_angle = delta_angle
            
        h_end_angle = h_start_angle + h_delta_angle
        v_end_angle = v_start_angle + v_delta_angle
        RotateCT(q_CT, 'h'+str(h_end_angle))
        RotateCT(q_CT, 'v'+str(v_end_angle))
        frame_end = GetPict(q_in, q_out)
        cv2.imwrite('frame_end.jpg',frame_end)
        h_rot = reco.SolveRotation(frame_start, frame_end, h_start_angle, h_end_angle)
        v_rot = reco.SolveRotation(frame_start, frame_end, v_start_angle, v_end_angle,Horizental=False)
        if h_rot==-1 or v_rot==-1:
            return False
        RotateCT(q_CT, 'h'+str(int(h_start_angle+h_rot)),sleep_time=0.1)
        RotateCT(q_CT, 'v'+str(int(v_start_angle+v_rot)))
        
    return valid



pCamera = Process(target = ShowVedio,args=(q_in, q_out, ))
#pCloudTable = Process(target = ControlCT,args=(q_CT, ))
pCamera.start()
#pCloudTable.start()



time.sleep(2)
q_in.put('target')
time.sleep(5)
q_in.put('laser')

time.sleep(2)
q_in.put('explode')
time.sleep(10)

#q_in.put('buyaotexiao')

'''
for i in range(5,1,-1):
    for j in range(7,2,-1):
        valid = LocateFace(25*i,15*j)
        if valid:
            break
    else:
        continue
    break


'''




pCamera.join()
#RotateCT(q_CT,'over')
#pCloudTable.join()













