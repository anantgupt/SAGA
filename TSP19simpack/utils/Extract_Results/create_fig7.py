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
def cf7(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_swidth-rob0/plot10.pickle','rb'))
    fig_handle2 = pl.load(open('fig_swidth-rob1/plot10.pickle','rb'))
    fig_handle3 = pl.load(open('fig_swidth-rob2/plot10.pickle','rb'))
    fig_handle4 = pl.load(open('fig_swidth-rob4/plot10.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    ax = fig_handle3.axes
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    mkr = [',','+','x','*']
    if 1:
        rng1 = fig_handle2.axes[1].lines[0].get_data()[0]
    #    crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        cnt = fig_handle3.axes[1].lines[1].get_data()[1]
        mse1 = fig_handle1.axes[1].lines[0].get_data()[1]
    
    #    crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2 = fig_handle2.axes[1].lines[0].get_data()[1]
    #    #cnt2 = fig_handle2.axes[3]
    #    crb3 = fig_handle3.axes[i].lines[1].get_data()[1]
        mse3 = fig_handle3.axes[1].lines[0].get_data()[1]
        mse4 = fig_handle4.axes[1].lines[0].get_data()[1]
    ax[1].clear()
    if 1:
        
    #    ax.plot(rng, mse1[i], 'b-', label='RMSE SNR=-10 dB')
        ax[1].plot(rng1, mse1, 'r-', marker=mkr[3], label=r'$\rho$=0')
        ax[1].plot(rng1, mse2, 'y-', marker=mkr[2], label=r'$\rho$=1')
        ax[1].plot(rng1, mse3, 'b-', marker=mkr[1], label=r'$\rho$=2')
        ax[1].plot(rng1, mse4, 'g-', marker=mkr[0], label=r'$\rho=4$')
        ax[1].plot(rng1, cnt, 'k--', label=r'True')
    #ax.plot(rng, mse4[0], 'b--s', label='Brute, Rob=1')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
    ax[1].legend(loc='best'),ax[1].grid(True);ax[0].set_xlabel('Iterations');
    #plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
    #plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
    #plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
    #plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
    ax[0].set_title('Graph Nodes v/s relax iterations');ax[0].set_ylabel('Num vertices')
    #ax[0].set_ylim(-2,80)
    
    # Subplot 2
    ax[1].set_title('Model order estimation');ax[1].set_ylabel('Num Targets detected')
    ax[1].set_xlabel(r'Array width $L_w$');ax[1].grid(True);ax[1].legend(loc='best')

    fig_handle3.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
        for text in axi.get_legend().get_texts(): text.set_fontsize(font_size)
    fig_handle3.set_tight_layout(True)
    pl.dump(fig_handle3, open("Sel_figs/plot_complex_vs_swidth-rob.pickle", "wb"))
    fig_handle3.savefig('Sel_figs/plot_complex_vs_swidth-rob.pdf')
#%%