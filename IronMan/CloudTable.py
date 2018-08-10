#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import serial

class CloudTable():
    # ser is the serial
    # lr_angle and tb_angle represent the horizental(left right) and vertical(top bottom) angle
    def __init__(self, device_name, lr_angle = 60, tb_angle = 30, sleep_time = 2):
        self.ser = serial.Serial(device_name,9600,timeout=5)
        self.lr_angle = lr_angle
        self.tb_angle = tb_angle
        self.sleep_time = sleep_time
        
        self.set_angle(lr_angle, tb_angle)
    
    def __del__(self):
        self.ser.close()
        del self.ser
        
    def set_angle(self, lr_angle=False,tb_angle=False):
        # if give a new lr_angle, the tell the ser to do the rotation
        if (lr_angle!=False and lr_angle!=self.lr_angle):
            self.ser.write(('h'+str(int(lr_angle))).encode())
            self.lr_angle = lr_angle
            time.sleep(self.sleep_time)
        
        # if give a new tb_angle, the tell the ser to do the rotation
        # need to change !!!!!!!!!!
        if (tb_angle!=False and tb_angle!=self.tb_angle):
            self.ser.write(('v'+str(int(tb_angle))).encode())
            self.tb_angle = tb_angle
            time.sleep(self.sleep_time)
            
    def set_light(self,signal):
        self.ser.write(signal.encode())