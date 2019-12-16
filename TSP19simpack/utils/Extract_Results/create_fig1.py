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
def cf1(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_snr-Nob1/plot2.pickle','rb'))
    fig_handle2 = pl.load(open('fig_snr-Nob10/plot2.pickle','rb'))
    fig_handle3 = pl.load(open('fig_snr-Nob20/plot2.pickle','rb'))
    fig_handle4 = pl.load(open('fig_snr-Nob30/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(1,2)
    for i in range(2):
        rng = fig_handle1.axes[i].lines[0].get_data()[0]
        crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        mse1 = fig_handle1.axes[i].lines[0].get_data()[1]
        #cnt1 = fig_handle1.axes[3]
        crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2 = fig_handle2.axes[i].lines[0].get_data()[1]
        #cnt2 = fig_handle2.axes[3]
        crb3 = fig_handle3.axes[i].lines[1].get_data()[1]
        mse3 = fig_handle3.axes[i].lines[0].get_data()[1]
        
        crb4 = fig_handle4.axes[i].lines[1].get_data()[1]
        mse4 = fig_handle4.axes[i].lines[0].get_data()[1]
        # ax[i].plot(rng, crb1, 'b--',label='CRB Nob=1')
        # ax[i].plot(rng, crb2, 'g--',label='CRB Nob=11')
        # ax[i].plot(rng, crb3, 'r--',label='CRB Nob=21')
        # ax[i].plot(rng, crb4, 'y--',label='CRB Nob=31')
        ax[i].plot(rng, crb1, 'b--')
        ax[i].plot(rng, crb2, 'g--')
        ax[i].plot(rng, crb3, 'r--')
        ax[i].plot(rng, crb4, 'y--')       
        ax[i].plot(rng, mse1, 'b-', marker =',', label='RMSE Nob=1')
        ax[i].plot(rng, mse2, 'g-',marker ='.', label='RMSE Nob=10')
        ax[i].plot(rng, mse3, 'r-', marker ='+',label='RMSE Nob=20')
        ax[i].plot(rng, mse4, 'y-', marker ='x',label='RMSE Nob=30')
        ax[i].legend(loc='lower left'),ax[i].grid(True);ax[i].set_xlabel('SNR (dB)');
    ax[0].set_title('Position Error');ax[0].set_ylabel('RMSE (dB) (m)')
    ax[1].set_title('Velocity Error');ax[1].set_ylabel('RMSE (dB) (m/s)')
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    #plt.yscale('log')
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
        for text in axi.get_legend().get_texts(): text.set_fontsize(font_size)
    fig.set_tight_layout(True)
    pl.dump(fig, open("Sel_figs/plot_PV_error.pickle", "wb"))
    fig.savefig('Sel_figs/plot_PV_error.pdf')