# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:22:37 2024

@author: human_lab
"""
from filters import filters
import numpy as np
import cv2
import utils
from numpy.fft import fft2, ifft2, fftshift, ifftshift

class CSP:
    def __init__(self, D , N, K, image):
        self.D, self.N, self.K= D, N, K
        self.image = image
        self.dim = self.image.ndim
        if self.dim == 3:
            self.H, self.W, _ = self.image.shape
        else:
            self.H, self.W = self.image.shape
        #filters
        self.HP = filters(self.N, self.K, self.H, self.W).HP()
        if self.dim != 2:
            self.HP = self.HP[:,:,None]
        self.bp_filters = []        
        self.get_bp_filters()
        #analysis 
        self.pyramid = []    #pyramid[depth][octave][orientation]        
        self.lp_res, self.hp_res = None, None
        self.decompose(image = self.image)
        
    def fft(self, image):
        if self.dim != 2:
            return fftshift(fft2(image, axes=(-3, -2)))
        elif self.dim == 2:
            return fftshift(fft2(image, axes=(-2, -1)))
        
    def ifft(self, image):
        if self.dim != 2:
            return ifft2(ifftshift(image), axes=(-3, -2))
        elif self.dim == 2:
            return ifft2(ifftshift(image), axes=(-2, -1))
        
    def get_bp_filters(self):
        for d in range(self.D):
            scale = np.power(2, d)
            h, w  = int(self.H/scale), int(self.W/scale)
            self.bp_filters.append(filters(self.N, self.K, h, w))
        
    def decompose(self, image):   
        print("decomp")
        #clear output
        self.pyramid.clear()
        
        image_dft = self.fft(image)
        
        #highpass residual
        self.hp_res = self.ifft(self.HP*image_dft)
 
        #bandpass
        Jt = image_dft
        for d in range(self.D):
            scale = np.power(2, d)
            h, w  = int(self.H/scale), int(self.W/scale)
            f = self.bp_filters[d]
            octaves = []
            for n in range(self.N):
                orientations = []
                for k in range(self.K):
                    BP = f.BP(n, k)
                    if self.dim != 2:
                        BP = BP[:,:,None]
                    P = self.ifft(Jt*BP)
                    orientations.append(P)
                octaves.append(orientations)
                
            self.pyramid.append(octaves)
            LP = f.LP()
            if self.dim != 2:
                LP = LP[:,:,None]
            J = self.ifft(Jt*LP)
            
            Jr = cv2.resize(J.real, (int(w/2), int(h/2)), interpolation = cv2.INTER_LINEAR)
            Ji = cv2.resize(J.imag, (int(w/2), int(h/2)), interpolation = cv2.INTER_LINEAR)
            Jt = self.fft(Jr + 1j*Ji)
              
        #lowpass residual
        self.lp_res = self.ifft(Jt)
        
        
    def reconstruct(self):
        print("recon")
        
        It = self.fft(self.lp_res)
        print(It.shape)
            
        for d in reversed(range(self.D)):
            I = self.ifft(It)
            if self.dim == 3:
                h, w, _ = I.shape
            else:
                h, w = I.shape
            
            Ir = cv2.resize(I.real, (2*w, 2*h), interpolation = cv2.INTER_LINEAR)
            Ii = cv2.resize(I.imag, (2*w, 2*h), interpolation = cv2.INTER_LINEAR)
            I = Ir + 1j *Ii
            LP = filters(self.N, self.K, 2*h, 2*w).LP()
            if self.dim != 2:
                LP = LP[:,:,None]
            It = self.fft(I)*LP

            for n in range(self.N):
                for k in range(self.K):
                    BP = self.bp_filters[d].BP(n, k)
                    if self.dim != 2:
                        BP = BP[:,:,None]
                    Jt = self.fft(self.pyramid[d][n][k]) * BP
                    #take conjugate and flip vertically and horizontally
                    Jt_conj = np.conjugate(Jt)
                    Jt_conj = np.flip(np.flip(Jt_conj , axis = 0), axis = 1)
                    It += 2*Jt #+ Jt_conj
 
        It += self.fft(self.hp_res)*self.HP
        return np.abs(self.ifft(It))
        
        