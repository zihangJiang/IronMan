#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np

FLIP = True


class Camera():
    def __init__(self, src=1, back_ground_path = 'operation_platform_pure_black.jpg'):
        self.capture = cv2.VideoCapture(src)
        self.src = src
        self.fore_ground = cv2.resize(cv2.imread(back_ground_path),(640,480))
        mask = self.fore_ground[:,:,0:1].copy()
        mask2 = self.fore_ground[:,:,1:2].copy()
        mask3 = self.fore_ground[:,:,2:3].copy()
        thresh = 4
        mask[mask<=thresh] = 0
        mask[mask>thresh] = 1
        mask2[mask2<=thresh] = 0
        mask2[mask2>thresh] = 1
        mask3[mask3<=thresh] = 0
        mask3[mask3>thresh] = 1
        self.mask = 1-(1-mask)*(1-mask2)*(1-mask3)
        # make sure the camera is ready
        assert self.capture.isOpened()
        self.video_writer = cv2.VideoWriter('jjj9898.mp4',cv2.VideoWriter_fourcc(*'XVID'),20,(640,480))
        
    
    def __del__(self):
        self.capture.release()
    
    
    def Refresh(self):
        self.capture.release()
        del self.capture
        self.capture = cv2.VideoCapture(self.src)
        # make sure the camera is ready
        assert self.capture.isOpened()
    
    
    def laser(self, frame):

        success, current_gif = self.gif.read()
        if success:
   
            scalex,scaley = (0.25,0.25)

            
            effe = cv2.resize(current_gif, (0, 0), fx=scalex, fy=scaley)
            h,w,_ = effe.shape
            effe = effe[:h-2,:w-2,:]
            h,w,_ = effe.shape
            ph,pw = (int((frame.shape[0]-h)),int((frame.shape[1]-w)/2))
            
            
            # 在pw，ph处放上特效

            mask = effe.copy()[:,:,:1]
            thresh = 210
            mask[mask<=thresh] = 0
            mask[mask>thresh] = 1
    
            
            frame[ph:ph+h,pw:pw+w,:] = frame[ph:ph+h,pw:pw+w,:] * (1-mask)
            frame[ph:ph+h,pw:pw+w,:] = frame[ph:ph+h,pw:pw+w,:] + effe*mask
            self.last_effe = effe
        else:
            #flag = 0
            #self.gif.release()
            #self.gif = cv2.VideoCapture(self.gif_file)
            effe = self.last_effe
            h,w,_ = effe.shape
            ph,pw = (int((frame.shape[0]-h)),int((frame.shape[1]-w)/2))
            
            mask = effe.copy()[:,:,:1]
            thresh = 200
            mask[mask<=thresh] = 0
            mask[mask>thresh] = 1
    
            
            frame[ph:ph+h,pw:pw+w,:] = frame[ph:ph+h,pw:pw+w,:] * (1-mask)
            frame[ph:ph+h,pw:pw+w,:] = frame[ph:ph+h,pw:pw+w,:] + effe*mask
            
        return True , frame

    def ShowVedio(self, q_in, q_out, effect_func = None):
        self.Refresh()
        while True:
            # Grab a single frame of video
            ret, frame = self.capture.read()
            if not ret:
                print('error')
                continue
            if FLIP:
                frame = cv2.flip(frame,1)
            if not q_in.empty():

                signal = q_in.get(True)
                if signal=='laizhangtu':
                    q_out.put(frame)
                    signal=None
                if signal=='laser':
                    signal=None
                    from EffectFunctions import AddLaser
                    effect_func = AddLaser

                if signal=='target':

                    signal=None
                    from EffectFunctions import AddTarget
                    effect_func = AddTarget
                if signal=='explode':
                    signal = None
                    from EffectFunctions import AddExplode
                    effect_func = AddExplode
                    
                if signal=='buyaotexiao':
                    signal==None
                    effect_func = None


            if effect_func!=None:
                handle, frame = effect_func(frame)
                if not handle:
                    effect_func = None
                    continue
                
            # Display the resulting image    

            frame = frame*(1-self.mask) + frame*self.mask*0.5 + self.fore_ground * self.mask * 0.5
            #frame = 0.5*self.fore_ground + 0.5*frame
            frame = frame.astype(np.uint8)
            #print(frame)
            cv2.imshow('Video', frame)
            self.video_writer.write(frame)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()
        
    def GetCurrentFrame(self):
        self.Refresh()
        ret, frame = self.capture.read()

        if ret:
            if FLIP:
                frame = cv2.flip(frame,1)
            return frame
        else:
            print('camera goes wrong')
            return None
        
    def SaveCurrentFrame(self, path = 'temp.jpg'):
        self.Refresh()
        ret, frame = self.capture.read()
        if ret:
            if FLIP:
                frame = cv2.flip(frame,1)
            cv2.imwrite(frame,path)
        else:
            print('camera goes wrong')
            
