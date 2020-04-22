# -*- coding: utf-8 -*-

import numpy as np

class RaoBallard1999Model:
    def __init__(self, dt=1, sigma2=1, sigma2_td=10):
        """
        Current-based Leaky integrate-and-fire model.
        
        Args:
            N (int)       : Number of neurons.
        """
        self.dt = dt
        self.inv_sigma2 = 1/sigma2 # 1 / sigma^2        
        self.inv_sigma2_td = 1/sigma2_td # 1 / sigma_td^2
        self.k1 = 0.5 # k_1: update rate
        self.k2 = 1 # k_2: learning rate
        self.lam = 0.02 # sparsity rate
        self.alpha1 = 1
        self.alpha2 = 0.05
        
        self.num_units_level0 = 256
        self.num_units_level1 = 32
        self.num_units_level2 = 64
        self.num_level1 = 3
        self.U = np.random.randn(self.num_level1,
                                 self.num_units_level0, 
                                 self.num_units_level1)
        self.Uh = np.random.randn(int(self.num_level1*self.num_units_level1),
                                  self.num_units_level2)
        self.r = np.random.rand(int(self.num_level1*self.num_units_level1))
        self.rh = np.random.rand(self.num_units_level2)
    
    def initialize_states(self, random_state=False):
        
    def __call__(self, inputs):
        # I : (3 x 256)
        r = np.reshape(self.r, (self.num_level1, self.num_units_level1))
        fx = np.array([np.tanh(self.U[i] @ r[i]) for i in range(self.num_level1)])
        fxh = np.tanh(self.Uh @ self.rh)
        dfx = 1 - fx**2
        dfxh = 1 - fxh**2
        error = inputs - fx
        error_h = self.r - fxh
        #dr = 
        #self.r += dr * self.dt
        #self.rh += drh * self.dt

        """
        drh = 
        dU =
        dUh = 
        np.tanh(x)
        """
        #for i in range(self.num_level1):
            
            
            