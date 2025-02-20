# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:25:20 2024

@author: human_lab
"""
import cv2
import utils
import numpy as np
from filters import filters
from pyramid import CSP
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from test_image import *
from skimage import data, color, img_as_float
import matplotlib.pyplot as plt

D, N, K, = 3, 2, 8

path = 'videos\guitar.avi'
vid,_ = utils.get_video(path, 1)
frame = vid[0]
#frame1 = vid[50]
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#frame = get_test_image(512, f1)
#frame = color.rgb2gray(np.repeat(data.brick()[:,:, None], 3, axis = 2))
frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)

if frame.ndim ==2:
    H, W = frame.shape
else:
    H, W, _ = frame.shape
    
f = filters(N, K, H, W)
b = f.BP(1,1)

s = f.HP()**2 + f.LP()**2 + sum([f.BP(n,k)**2 for n in range(N) for k in range(K)])
#utils.imshow(s)

#pyramid
csp = CSP(D , N, K, frame)
d = 0
for n in range(0):
    utils.imshow(cv2.normalize(f.W(n, f.radial), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
    for k in range(K):
        d, n = 0, 0
        #utils.imshow(cv2.normalize(f.G(k, f.angular), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1))
        utils.imshow(csp.bp_filters[d].BP(n, k))
        #utils.imshow(csp.pyramid[d][n][k])
    
    
recon = csp.reconstruct()
recon = np.clip(recon, 0, 255)#.astype(np.uint8)
#print(recon)
utils.imshow(frame)
utils.imshow(recon)
plt.imshow((frame - recon))
print(np.mean(np.square(recon - frame)))

dft = fftshift(fft2(frame))
recond = ifft2(ifftshift(dft))
