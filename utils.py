# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:16:53 2024

@author: human_lab
"""
import cv2
import numpy as np

def get_video(video_path, scale = 1):
    
    vid = cv2.VideoCapture(video_path)
    fs = vid.get(cv2.CAP_PROP_FPS)
    frames = []
    
    idx = 0
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        if idx ==0:
            h, w, _ = frame.shape
            h, w = scale*h, scale*w
        frames.append(cv2.resize(frame, (w,h)))
        idx +=1
        
    vid.release()
    cv2.destroyAllWindows()
    del vid
        
    return frames, fs

        
def imshow(img):
    cv2.namedWindow("image")
    cv2.imshow("image", cv2.normalize(np.abs(img), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    cv2.waitKey()
    cv2.destroyAllWindows()

def norm(image):
    scaling = image.max() - image.min()
    if scaling == 0:
        return image
    return (image - image.min())/(scaling)
    