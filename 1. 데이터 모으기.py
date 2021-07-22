#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random
import pathlib
import shutil


# In[2]:


cap = cv2.VideoCapture(0)

params= list()
params.append(cv2.IMWRITE_PNG_COMPRESSION)
params.append(8)
cwd = os.getcwd()


# In[3]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[5]:


i = input()
if not os.path.isdir("C:/dev-tf/facedata/" + i):
    os.makedirs('C:/dev-tf/facedata/' + i)
if not os.path.isdir("C:/dev-tf/facedata/test/" + i):
    os.makedirs("C:/dev-tf/facedata/test/" + i)
if not os.path.isdir("C:/dev-tf/facedata/train/" + i):
    os.makedirs("C:/dev-tf/facedata/train/" + i)
if not os.path.isdir("C:/dev-tf/facedata/valid/" + i):
    os.makedirs("C:/dev-tf/facedata/valid/" + i)

while True:
    ret, frame = cap.read();
    if not ret: continue
    rows, cols, channels = frame.shape
    width = cols
    height = rows
    length = min(width, height)
    pt = [60,60] 
    if width < height:
        pt[1] += int((height - length) / 2)
    else:
        pt[0] += int((width - length) / 2)
    green = (0, 255, 0)
    length -= 120   
    cv2.rectangle(frame, (pt[0], pt[1]), (pt[0]+length, pt[1]+length), green, 4)
    cv2.imshow('view', frame)
    ch = cv2.waitKey(1) & 0xFF
    if ch == 27:
        cv2.destroyAllWindows()
        break
    if ch == 32:
        mid_frame = frame[pt[1]:pt[1]+length, pt[0]:pt[0]+length]
        cut_frame = cv2.resize(mid_frame, (224,224))
        cv2.imshow('cut', cut_frame)
        filename = 'C:/dev-tf/facedata/' + i + '/' +strftime('%d%H%M%S', localtime())+'.png'
        cv2.imwrite(filename, cut_frame, params)
    
file_list = os.listdir('C:/dev-tf/facedata/' + i)
# random.shuffle(file_list)
cnt = 1
for file_name in file_list:
    try:
        if cnt % 5 == 3:
            shutil.move('C:/dev-tf/facedata/' + i + '/' + file_name, 'C:/dev-tf/facedata/test/' + i + '/' + file_name)
        if cnt % 5 == 4:
            shutil.move('C:/dev-tf/facedata/' + i + '/' + file_name, 'C:/dev-tf/facedata/valid/' + i + '/' + file_name)
        else:
            shutil.move('C:/dev-tf/facedata/' + i + '/' + file_name, 'C:/dev-tf/facedata/train/' + i + '/' + file_name)
    except:
        print('저장 실패')
    cnt+= 1


# In[ ]:




