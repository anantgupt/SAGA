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
def cf8(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_Nob-Nsens4/plot11.pickle','rb'))
    fig_handle2 = pl.load(open('fig_Nob-Nsens6/plot11.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(1,1)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    lbl=[" All edges"," Brute"," "+mode]
    mkr = [',','+','x','*']
    indxa = [1,2]
    for i in range(3):
        rng1 = fig_handle2.axes[1].lines[i].get_data()[0]
        mse1[i] = fig_handle1.axes[1].lines[i].get_data()[1]
    
    #    crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2[i] = fig_handle2.axes[1].lines[i].get_data()[1]
    for i in indxa:
    #    ax.plot(rng, mse1[i], 'b-', label='RMSE SNR=-10 dB')
        ax.plot(rng1, mse1[i], 'r-', marker=mkr[i], label=r'$N_{sensors}$=4'+lbl[i])
    
        ax.plot(rng1, mse2[i], 'b-.', marker=mkr[i], label=r'$N_{sensors}$=6'+lbl[i])
    #    ax[1].plot(rng1, cnt, 'k--', label=r'True')
    #ax.plot(rng, mse4[0], 'b--s', label='Brute, Rob=1')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('Number of targets');
    #plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
    #plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
    #plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
    #plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
    ax.set_title('Association Complexity');ax.set_ylabel('Tracks visited')

    ax.set_yscale('log')

    
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.tight_layout()
    pl.dump(fig, open("Sel_figs/plot_Complex-Nob-Nsens.pickle", "wb"))
    fig.savefig('Sel_figs/plot_Complex-Nob-Nsens.pdf')
#%%