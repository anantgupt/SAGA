#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:57:27 2019
Use this code to edit figures saved using pickle dump
@author: anantgupta
"""

import matplotlib.pyplot as plt
import pickle as pl
import numpy as np

# Load figure from disk and display

#fig_handle1 = pl.load(open('fig_Nsens-swidth0.4/plot10.pickle','rb'))
fig_handle1 = pl.load(open('fig_snr-Nob10_Abs/plot3.pickle','rb'))
fig_handle2 = pl.load(open('fig_snr-Nob10_Rel/plot3.pickle','rb'))

#fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))

fig, ax = plt.subplots(1,1)
mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
lbl=[" All edges"," Brute"," Relax"]
mkr = [',','+','x','*']
indxa = [1,2]
rng1 = (fig_handle2.axes[1].lines[0].get_data()[0]) #Max-Min

mse1 = fig_handle1.axes[1].lines[0].get_data()[1]
mse2 = fig_handle2.axes[1].lines[0].get_data()[1]
ax.plot(rng1, mse1, 'r-', label='Abs CFAR')
ax.plot(rng1, mse2, 'b-s', label='Relative')


ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('SNR (dB)');
#plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
#plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
#plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
#plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
#ax.set_title('Association Complexity');ax.set_ylabel('Tracks visited')
#ax[0].set_ylim(-2,80)

ax.set_title('Localization Error');ax.set_ylabel('RMSE Error')

pl.dump(fig, open("plot_Abs_Rel-snr.pickle", "wb"))
#%%