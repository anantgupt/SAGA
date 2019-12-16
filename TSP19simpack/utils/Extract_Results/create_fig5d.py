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
def cf5d(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_Nob-Nsens4/plot3.pickle','rb'))
    fig_handle2 = pl.load(open('fig_Nob-Nsens6/plot3.pickle','rb'))

    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(2,1)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    for i in range(2):
        rng = fig_handle1.axes[i*2].lines[0].get_data()[0]
        mse1[i] = fig_handle1.axes[i*2].lines[0].get_data()[1]
        mse2[i] = fig_handle2.axes[i*2].lines[0].get_data()[1]
      
        
        ax[i].plot(rng, mse1[i], 'g-', label=mode+r', $N_s=4$')
    
        ax[i].plot(rng, mse2[i], 'g-.', label=mode+r', $N_s=6$')


        ax[i].legend(loc='best'),ax[i].grid(True);ax[i].set_xlabel(fig_handle1.axes[2*i].get_xlabel());
        ax[i].set_title(fig_handle1.axes[2*i].get_title());ax[i].set_ylabel(fig_handle1.axes[2*i].get_ylabel())
    #ax[1].set_title('Localization Error');ax[1].set_ylabel('RMSE (dB)')
    #ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
    plt.tight_layout()
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    ax[0].set_yscale('log')
    ax[1].set_yscale('symlog')
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
        for text in axi.get_legend().get_texts(): text.set_fontsize(font_size)
    
    fig.set_tight_layout(True)
    pl.dump(fig, open("Sel_figs/plot_OSPA_vs_Nob-Nsens.pickle", "wb"))
    fig.savefig('Sel_figs/plot_OSPA_vs_Nob-Nsens.pdf')