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

#fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
fig_handle1 = pl.load(open('DFT/fig_snr-Nob21/plot3.pickle','rb'))
fig_handle2 = pl.load(open('DFT/fig_snr-Nob21/plot1.pickle','rb'))
fig_handle3 = pl.load(open('NOMP/fig_snr-Nob21/plot3.pickle','rb'))
fig_handle4 = pl.load(open('NOMP/fig_snr-Nob21/plot1.pickle','rb'))

#fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))

fig, ax = plt.subplots(1,1)
mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
l1=[0]*2; l2 =[0]*2; l3 = [0]*2
lbl = ['Estimation time','Association time']
mkr = [',','x']
for i in range(2):
    rng = fig_handle2.axes[1].lines[i*2].get_data()[0]
    c1 = fig_handle2.axes[1].lines[i*2].get_data()[1]
    c2 = fig_handle4.axes[1].lines[i*2].get_data()[1]


#    mse1[i] = fig_handle1.axes[1].lines[i].get_data()[1]
    #cnt1 = fig_handle1.axes[3]
#    crb2[i] = fig_handle2.axes[i].lines[1].get_data()[1]
    
    l1[i] = ax.plot(rng, c1, 'r', marker =mkr[i], label=lbl[i]+', DFT')[0]
    l2[i] = ax.plot(rng, c2, 'b', marker = mkr[i], label=lbl[i]+', NOMP')[0]
#    ax[i].plot(rng, mse2[i], 'r-.', label='DFT')
ax2 = ax.twinx()
c1b = fig_handle1.axes[0].lines[0].get_data()[1]
c2b = fig_handle3.axes[0].lines[0].get_data()[1]
l3[0] = ax2.plot(rng, c1b, 'r-.s', label = 'DFT OSPA')[0]
l3[1] = ax2.plot(rng, c2b, 'b-.s', label = 'NOMP OSPA')[0]
ax2.set_ylabel('OSPA')
ax2.set_ylim(0,1.2)
ax.set_ylim(0,2100)
#    ax.plot(rng, mse1[i], 'b-', label='RMSE SNR=-10 dB')
#ax.plot(rng, crb2, 'k--', label='CRB')
#ax.plot(rng, mse2[0], 'r-.', label='DFT')
#ax.plot(rng, mse4[0], 'b-', label='NOMP')
#ax.plot(rng, mse4[0], 'b--s', label='Brute, Rob=1')
#    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
ax2.legend(l1+l2+l3, [l1[0]._label, l1[1]._label,l2[0]._label,l2[1]._label, l3[0]._label,l3[1]._label],loc='upper center')
ax.grid(True);ax.set_xlabel('SNR');ax.set_ylabel('Runtime (s)');
#plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
#plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
#plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
#plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
#ax.set_title('Association Complexity');ax.set_ylabel('Number of tracks visited')
ax.set_title('Runtime comparison')
#ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
plt.tight_layout()
#plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
#ax.set_yscale('log')
pl.dump(fig_handle2, open("plot_Runtime_vs_snr.pickle", "wb"))

#%%
#plt.figure(2)
#fig_handle = pl.load(open('res_comb/plot_doppler_resolution.pickle','rb'))