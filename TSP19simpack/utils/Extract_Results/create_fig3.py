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
def cf3(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_Nob-snr-10/plot3.pickle','rb'))
    fig_handle2 = pl.load(open('fig_Nob-snr-15/plot3.pickle','rb'))
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(1,3)
    for i in range(3):
        rng = fig_handle1.axes[i].lines[0].get_data()[0]
    #    crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        mse1 = fig_handle1.axes[i].lines[0].get_data()[1]
        #cnt1 = fig_handle1.axes[3]
    #    crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2 = fig_handle2.axes[i].lines[0].get_data()[1]
    #    #cnt2 = fig_handle2.axes[3]
    #    crb3 = fig_handle3.axes[i].lines[1].get_data()[1]
    #    mse3 = fig_handle3.axes[i].lines[0].get_data()[1]
        #cnt3 = fig_handle3.axes[3]
    #for i in range(3):
    #    for line in fig_handle2.axes[i].lines[:10]:
    #        mser[t]=line.get_data()[1]
    #        t+=1
    #    ax[i].plot(rng, crb1, 'b--',label='CRB SNR=-10 dB')
    #    ax[i].plot(rng, crb2, 'g--',label='CRB Nob=-15')
    #    ax[i].plot(rng, crb3, 'r--',label='CRB Nob=21')
        
        ax[i].plot(rng, mse1, 'b-', label='RMSE SNR=-10 dB')
        ax[i].plot(rng, mse2, 'g-', label='RMSE SNR=-15 dB')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
        ax[i].legend(loc='best'),ax[i].grid(True);ax[i].set_xlabel('Num Targets');
    #plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
    #plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
    #plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
    #plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
    ax[0].set_title('OSPA');ax[0].set_ylabel('OSPA (dB) (m)')
    ax[1].set_title('Localization Error');ax[1].set_ylabel('RMSE (dB)')
    ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    ax[0].set_yscale('log');ax[1].set_yscale('log')
#    ax[0].set_ylim(0.1,10);
#    ax[1].set_ylim(0.1,10);
    
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
    plt.tight_layout()

    pl.dump(plt.figure(1), open("Sel_figs/plot_OSPA_error_Nob.pickle", "wb"))
    fig.savefig('Sel_figs/plot_OSPA_error_Nob.pdf', Transparent=True)
    #%%
    #plt.figure(2)
    #fig_handle = pl.load(open('res_comb/plot_doppler_resolution.pickle','rb'))