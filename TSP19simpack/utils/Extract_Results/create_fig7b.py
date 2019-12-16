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
def cf7b(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_swidth-rob0/plot7.pickle','rb'))
    fig_handle2 = pl.load(open('fig_swidth-rob1/plot7.pickle','rb'))
    fig_handle3 = pl.load(open('fig_swidth-rob2/plot7.pickle','rb'))
    fig_handle4 = pl.load(open('fig_swidth-rob4/plot7.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(1,1)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    lbl=[" Detection","False Alarm"," Miss"]
    mkr = [',','+','x','*']
    indxa = [1,2]
    for i in range(3):
        rng1 = fig_handle2.axes[0].lines[i].get_data()[0]
        mse1[i] = fig_handle1.axes[0].lines[i].get_data()[1]
    
    #    crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2[i] = fig_handle2.axes[0].lines[i].get_data()[1]
    #    #cnt2 = fig_handle2.axes[3]
    #    crb3 = fig_handle3.axes[i].lines[1].get_data()[1]
        mse3[i] = fig_handle3.axes[0].lines[i].get_data()[1]
        mse4[i] = fig_handle4.axes[0].lines[i].get_data()[1]
    for i in indxa:
    #    ax.plot(rng, mse1[i], 'b-', label='RMSE SNR=-10 dB')
        ax.plot(rng1, mse1[i], 'r-', marker=mkr[i], label=r'$\rho$=0'+lbl[i])
    
        ax.plot(rng1, mse2[i], 'y-', marker=mkr[i], label=r'$\rho$=1'+lbl[i])
        ax.plot(rng1, mse3[i], 'b-', marker=mkr[i], label=r'$\rho$=2'+lbl[i])
        ax.plot(rng1, mse4[i], 'g-', marker=mkr[i], label=r'$\rho=4$'+lbl[i])
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('Array Width');
    ax.set_title(r'$P_{False Alarm}, P_{miss}$');ax.set_ylabel(r'$P_{FA}/ P_{miss}$')

    
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.tight_layout()
    pl.dump(fig, open("Sel_figs/plot_PD-miss.pickle", "wb"))
    fig.savefig('Sel_figs/plot_PD-miss.pdf')
    #%%