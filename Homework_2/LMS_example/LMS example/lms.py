# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 21:11:47 2019

@author: arindam
"""
import numpy as np
from scipy import signal
import pdb

class LMS():
    def __init__(self, L=10, step=0.1):
        self.L = L
        self.step = step
        self.w = np.zeros(self.L)
        self.eps = 1e-3
    
    def __create_arrays__(self, x):
        v = np.zeros((len(x), self.L))
        v[0,0] = x[0]
        for i in range(1,len(x)):
            v[i,1:] = v[i-1, 0:-1]
            v[i,0] = x[i]
        return v
    def adapt(self, x, d, N=100, normalize=True):
        assert len(x)==len(d)
        v = self.__create_arrays__(x)
        N = len(d)
        for i in range(N):
            prod = np.inner(self.w, v[i,:])
            if normalize:
                self.w += self.step * (d[i]-prod) * v[i] / (np.inner(v[i],v[i])+self.eps)
            else:
                self.w += self.step * (d[i]-prod) * v[i]
        
        y = signal.lfilter(self.w, 1, x)
        e = d - y
        print('LMS successfully adapted the weights!')
        return e, y, self.w
        
            
        
        
        
        
