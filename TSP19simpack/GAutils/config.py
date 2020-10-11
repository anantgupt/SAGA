#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:57:28 2019
Configuration parameters for tracking_main
@author: anantgupta
"""
import numpy as np

## Fixed
debug_plots = False # Debugging plots
movie = False # Generate MOvie of simulation (Requires FFMPEG)
scene_plots = False

Nf =5
Ninst=4
parallel = True
sensor_wise = True # Results display style
max_sensors = 20 # Set based on script4 max value

N_cpu = -1 # Set Num of parallel cores automatically

# Scene simulation iterabes
roba = 1*np.ones(Ninst) # Robustness level (Should be less than N_sens-1)
snra = -10*np.ones(Ninst)
Nsensa = 4*np.ones(Ninst, dtype='int')
Noba = 10*np.ones(Ninst, dtype='int')  # number of targets
swidtha = 4*np.ones(Ninst, dtype='int') # Array width
pmissa=0.05*np.ones(Ninst) # Miss probability
# Iterable descriptors
rng_used = Nsensa
xlbl = 'Num sensors' #'SNR (dB)' # 'Num objects', 'Num sensors'
# Scalars
sep_th = 1 # Separation threshold
rd_wt = [1,1] # Range doppler relative weighting for likelihood, NLLS (Selection purposes)
all_pht = True # All possible pht or only consecutive pairs
static_snapshot = 1
colr=['r','b','g']
#Estimation 
estalgo=2 #0:FFT, 1:FFTintp,2:NOMP 
osps = [3,3] # 2 is good enuf
n_Rc = [1,2] # Test with 2,3
n_pfa = 1e-2 # Optimal is 1e-2 or 2e-2
# Association
mode = 'SAGA' # Choose: 'SAGA','SAESL','NN','MCF' Old:'Brute','DFS','Relax','Brute_iter','mcf','SPEKF','Relax-heap','SPEKF-heap'
#crb_min =np.array([1e-2, 1e-2]) # Empirical variance of Range, Doppler (For LLR)
hscale= np.array([2,2]) # Growth in thresold (squared for fitting threshold)
incr = 2 # Increment in association iterations
hN = 20 # Relaxation iterations
ag_pfa = 5e-2 # Sets beginning value of geometric fitting thres 
al_pfa = 5e-2 # Sets beginning value of likelihood thres
Tlen = 3 #was 3, Ns- rob At least Tlen nodes in track
# Gauss Newton
fu_alg = 'ls' # Least-Square: 'ls', Huber: 'huber', 'l1','l2': Same as ls but usins cvx
gn_steps = 5
folder = 'rawfigs2'