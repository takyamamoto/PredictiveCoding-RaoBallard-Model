# -*- coding: utf-8 -*-

import numpy as np

class RaoBallard1999Model:
    def __init__(self, dt=1e-3, sigma2=1, sigma2_td=10):
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
        self.alpha = 1
        self.alphah = 0.05
        
        self.num_units_level0 = 256
        self.num_units_level1 = 32
        self.num_units_level2 = 64
        self.num_level1 = 3
        
        self.U = np.random.randn(self.num_level1,
                                 self.num_units_level0, 
                                 self.num_units_level1)
        self.Uh = np.random.randn(int(self.num_level1*self.num_units_level1),
                                  self.num_units_level2)
        self.r = np.random.rand(self.num_level1, self.num_units_level1)
        self.rh = np.random.rand(self.num_units_level2)
    
    def initialize_states(self):
        self.r = np.random.rand(self.num_level1, self.num_units_level1)
        self.rh = np.random.rand(self.num_units_level2)
        
    def __call__(self, inputs, training=True):
        # inputs : (3, 256)
        r_reshaped = np.reshape(self.r, (int(self.num_level1*self.num_units_level1))) # (96)

        fx = np.array([np.tanh(self.U[i] @ self.r[i]) for i in range(self.num_level1)]) # (3, 256)
        fxh = np.tanh(self.Uh @ self.rh) # (96, )
        
        dfx = 1 - fx**2 # (3, 256)
        dfxh = 1 - fxh**2 # (96,)
        
        error = inputs - fx # (3, 256)
        errorh = r_reshaped - fxh # (96, ) 
        errorh_reshaped = np.reshape(errorh, (self.num_level1, self.num_units_level1)) # (3, 32)

        dfx_error = dfx * error # (3, 256)
        dfxh_errorh = dfxh * errorh # (96, )
        
        g_r = self.alpha * self.r / (1 + self.r**2) # (3, 32)
        g_rh = self.alphah * self.rh / (1 + self.rh**2) # (64, )
        
        dr = self.inv_sigma2 * np.array([self.U[i].T @ dfx_error[i] for i in range(self.num_level1)])\
            - self.inv_sigma2_td * errorh_reshaped - g_r
        drh = self.inv_sigma2_td * self.Uh.T @ dfxh_errorh - g_rh

        if training:            
            dU = self.inv_sigma2 * np.array([np.outer(dfx_error[i], self.r[i]) for i in range(self.num_level1)])\
                - self.lam * self.U
            dUh = self.inv_sigma2_td * np.outer(dfxh_errorh, self.rh)\
                - self.lam * self.Uh

        self.r += self.k1 * dr * self.dt
        self.rh += self.k1 * drh * self.dt
        self.U += self.k2 * dU * self.dt
        self.Uh += self.k2 * dUh * self.dt
        return error
            
            
            