# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:54:23 2024

@author: human_lab
"""
import numpy as np
import scipy.fftpack
from numpy.fft import fft, ifft, fftshift, ifftshift

class filters():
    def __init__(self, N, K, H, W, center = None):
        self.N = N  #number of filters per octave
        self.K = K  #number of filters per orientation
        self.h = H  #image height
        self.w = W  #image width
        self.center = center
        if self.center is None:
            self.center = (int(self.h/2), int(self.w/2))
        self.size = max(self.h, self.w)
        
        ''' 
        self.Y, self.X = np.ogrid[:self.h, :self.w]                         #grid
        self.X, self.Y = self.X - self.center[1], self.center[0] - self.Y   #shift to center 
        self.epsilon = 0
        self.radial = np.pi*np.sqrt(self.X**2 + self.Y**2)/(self.size)
        self.radial = np.where(self.radial < 1e-10, 1e-10, self.radial)
        self.angular = np.arctan2(self.Y, self.X)
        '''
        d_y =  1/(2.*np.pi*self.h/self.size)
        d_x =  1/(2.*np.pi*self.w/self.size)
        w_y = scipy.fftpack.fftfreq(self.h,d=d_y)
        w_x = scipy.fftpack.fftfreq(self.w,d=d_x)
        W = np.stack((np.repeat(w_y.reshape(-1,1),self.w,axis=1),np.repeat(w_x.reshape(1,-1),self.h,axis=0)))
        self.radial = np.linalg.norm(W,axis=0)
        self.radial = np.where(self.radial <= 0, 1e-40, self.radial)
        self.angular = np.arctan2(W[0],W[1])
        
        
    def L(self, r):                                  # low pass
        return  np.where( r < np.pi/4, 1, np.cos((np.pi/2)*np.log2(4*r/np.pi)) ) * (r <= np.pi/2) 

    def H(self, r):                                  # high pass
        return  np.where( r > np.pi/2, 1, np.cos((np.pi/2)*np.log2(2*r/np.pi)) ) * (r >= np.pi/4) 
        
    def G(self, k, theta):                           # band pass angular
        factor = np.math.factorial(self.K - 1)/np.sqrt(self.K*np.math.factorial(2*self.K - 2))
        theta = np.mod(theta + np.pi - np.pi*(k/(self.K )), 2*np.pi) - np.pi  #unwrap
        
        return np.where(np.abs(theta) < np.pi/2, factor*np.power(2*np.cos(theta), self.K - 1), 0)
        
    def W(self, n, r):                               # band pass radial
        return self.H(r)*self.L(r)
        #return self.H(r/np.power(2, (self.N - n)/self.N)) * self.L(r/np.power(2, (self.N - n + 1)/self.N)) 
    
    def BP(self, n, k):                              # band pass 
        return self.W(n, self.radial) * self.G(k, self.angular)
    
    def LP(self):
        return self.L(self.radial)
     
    def HP(self):
        return self.H(self.radial)
    
class temporal_filter():
    
    def __init__(self, freq_lo, freq_hi, freq_sample, length):
        self.freq_lo, self.freq_hi, self.freq_sample, self.length = freq_lo, freq_hi, freq_sample, length
        self.F = None
        if self.freq_hi >= self.freq_sample/2.:
            self.F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(self.length,self.freq_lo,fs=self.freq_sample,pass_zero=False)))
        else:
            self.F = scipy.fftpack.fft(scipy.fftpack.ifftshift(scipy.signal.firwin(self.length,[self.freq_lo,self.freq_hi],fs=self.freq_sample,pass_zero=False)))
        

    def tempFilter(self, frames):
        self.h, self.w, self.c = frames[0].shape
        self.frame_array = np.zeros([self.h, self.w, self.c, len(frames)])
        for i, frame in enumerate(frames):
            self.frame_array[:,:,:,i] = frame
        self.return_array = np.zeros_like(self.frame_array)

        for c in range(self.c):
            for h in range(self.h):
                for w in range(self.w):
                    x = self.frame_array[h, w, c, :]
                    y = fft(x)*self.F
                    self.return_array[h, w, c, :] = ifft(y)
                    
        #self.frame_array = np.squeeze(self.frame_array)
        #self.return_array = np.squeeze(self.return_array)
        if self.c == 1:
            return [self.return_array[:,:,i] for i in range(self.return_array.shape[-1])]
        else:
            return [self.return_array[:,:,:,i] for i in range(self.return_array.shape[-1])]
    
    
    
    