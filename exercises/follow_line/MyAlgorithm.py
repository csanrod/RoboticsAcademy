#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
from datetime import datetime

import math
import cv2
import numpy as np

time_cycle = 80

class MyAlgorithm(threading.Thread):

    def __init__(self, camera, motors):
        self.camera = camera
        self.motors = motors
        self.threshold_image = np.zeros((640,360,3), np.uint8)
        self.color_image = np.zeros((640,360,3), np.uint8)
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
    
    def getImage(self):
        self.lock.acquire()
        img = self.camera.getImage().data
        self.lock.release()
        return img

    def set_color_image (self, image):
        img  = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.color_image_lock.acquire()
        self.color_image = img
        self.color_image_lock.release()
        
    def get_color_image (self):
        self.color_image_lock.acquire()
        img = np.copy(self.color_image)
        self.color_image_lock.release()
        return img
        
    def set_threshold_image (self, image):
        img = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.threshold_image_lock.acquire()
        self.threshold_image = img
        self.threshold_image_lock.release()
        
    def get_threshold_image (self):
        self.threshold_image_lock.acquire()
        img  = np.copy(self.threshold_image)
        self.threshold_image_lock.release()
        return img

    def run (self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.algorithm()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):
        image = self.getImage()

        #Convert RGB --> HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lower = np.array([0, 70, 50], dtype = "uint8")
        upper = np.array([10, 255, 255], dtype = "uint8")

        mask = cv2.inRange(hsv_image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        #for row in range (250, 400):
        #    output[row] = np.where(output[row]!=[0,0,0],[255,255,255],output[row])
            
        top_indx_avg = 0
        bot_indx_avg = 0

        count = 0
        arr_indx = np.where(output[380] != [0,0,0])
        for i in arr_indx[0]:
            bot_indx_avg += i
            count += 1
        bot_indx_avg /= count

        count = 0
        arr_indx = np.where(output[280] != [0,0,0])
        for i in arr_indx[0]:
            top_indx_avg += i
            count += 1
        top_indx_avg /= count
        
        #Controlador P
        #self.motors.sendV(10)
        #self.motors.sendW(-5)
        R = 1.0
        Y = top_indx_avg - bot_indx_avg
        err = R - Y

        print Y
        print err

        Kp = 0.55 #100%/Max err

        W_max = 1.0
        control = (-Kp*err)/100 # % de acción sobre W
        print("control")
        print control
        print("---")

        self.motors.sendW(W_max*control)
        self.motors.sendV(3)


        self.set_threshold_image(output)
