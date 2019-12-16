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
def cf6b(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_rob-swidth0_2/plot10.pickle','rb'))
    fig_handle2 = pl.load(open('fig_rob-swidth0_5/plot10.pickle','rb'))
    fig_handle3 = pl.load(open('fig_rob-swidth1/plot10.pickle','rb'))
    #fig_handle4 = pl.load(open('fig_rob-swidth0.8/plot3.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(2,1)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    mkr = [',','+','x']
    for i in range(3):
        rng11 = fig_handle1.axes[0].lines[i].get_data()[0]
        rng2 = fig_handle2.axes[1].lines[0].get_data()[0]
    #    crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        mse1[i] = fig_handle1.axes[0].lines[i].get_data()[1]
    
        rng22 = fig_handle2.axes[0].lines[i].get_data()[0]
        mse2[i] = fig_handle2.axes[0].lines[i].get_data()[1]
    #    #cnt2 = fig_handle2.axes[3]
    #    crb3 = fig_handle3.axes[i].lines[1].get_data()[1]
        rng33 = fig_handle3.axes[0].lines[i].get_data()[0]
        mse3[i] = fig_handle3.axes[0].lines[i].get_data()[1]
    
        
    #    ax.plot(rng, mse1[i], 'b-', label='RMSE SNR=-10 dB')
        ax[0].plot(rng11, mse1[i], 'r-', marker=mkr[i], label=r'$\rho$='+str(i)+', $L_w$=0.2m')
        ax[0].plot(rng22, mse2[i], 'b-', marker=mkr[i], label=r'$\rho$='+str(i)+', $L_w$=0.5m')
        ax[0].plot(rng33, mse3[i], 'g-', marker=mkr[i], label=r'$\rho$='+str(i)+', $L_w$=1m')
    #ax.plot(rng, mse4[0], 'b--s', label='Brute, Rob=1')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
    ax[0].legend(loc='best'),ax[0].grid(True);ax[0].set_xlabel('Iterations');
    #plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
    #plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
    #plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
    #plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
    ax[0].set_title('Graph Nodes v/s '+mode+' iterations');ax[0].set_ylabel('Num vertices')
#    ax[0].set_ylim(-2,80)
    
    # Subplot 2
    cnt = fig_handle1.axes[1].lines[1].get_data()[1]
    cnt1 = fig_handle1.axes[1].lines[0].get_data()[1]
    cnt2 = fig_handle2.axes[1].lines[0].get_data()[1]
    cnt3 = fig_handle3.axes[1].lines[0].get_data()[1]
    ax[1].plot(rng2, cnt1, 'r-', label=r'$L_w$=0.2m')
    ax[1].plot(rng2, cnt2, 'b-o', label=r'$L_w$=0.5m')
    ax[1].plot(rng2, cnt3, 'g-s', label=r'$L_w$=1m')
    ax[1].plot(rng2, cnt, 'k--', label='True')
    ax[1].set_title('Number of targets detected');ax[1].set_ylabel('Num Targets detected')
    ax[1].set_xlabel('Robustness level');ax[1].grid(True);ax[1].legend(loc='best')
    #ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
    
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
        for text in axi.get_legend().get_texts(): text.set_fontsize(font_size)
    fig.set_tight_layout(True)
    pl.dump(fig, open("Sel_figs/plot_complex_vs_rob-swidth.pickle", "wb"))
    fig.savefig('Sel_figs/plot_complex_vs_rob-swidth.pdf')

#%%