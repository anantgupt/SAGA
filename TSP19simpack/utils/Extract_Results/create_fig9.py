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
def cf9(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle1 = pl.load(open('fig_Nsens-swidth0.4/plot10.pickle','rb'))
    fig_handle1 = pl.load(open('fig_swidth-Nsens4/plot3.pickle','rb'))
    fig_handle2 = pl.load(open('fig_swidth-Nsens8/plot3.pickle','rb'))
    # fig_handle3 = pl.load(open('fig_swidth-Nsens12/plot3.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, ax = plt.subplots(1,3)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    lbl=[" All edges"," Brute",mode]
    mkr = [',','+','x','*']
    indxa = [1,2]
    for i in range(3):
        rng1 = (fig_handle2.axes[i].lines[0].get_data()[0]) #Max-Min
        rng1 = [2*r for r in rng1] 
        mse1[i] = fig_handle1.axes[i].lines[0].get_data()[1]
    
    #    crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2[i] = fig_handle2.axes[i].lines[0].get_data()[1]
    
        ax[i].plot(rng1, mse1[i], 'r-', label=r'$N_{sensors}$=4')
    
        ax[i].plot(rng1, mse2[i], 'b-.', label=r'$N_{sensors}$=8')
    #    ax[1].plot(rng1, cnt, 'k--', label=r'True')
    #ax.plot(rng, mse4[0], 'b--s', label='Brute, Rob=1')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
        ax[i].legend(loc='best'),ax[i].grid(True);ax[i].set_xlabel('Array width $L_w$ (m)');
    
    ax[0].set_title('OSPA');ax[0].set_ylabel('OSPA (dB) (m)')
    ax[1].set_title('Localization Error');ax[1].set_ylabel('RMSE (dB)')
    ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    #plt.yscale('log')
    ax[0].set_yscale('log');ax[1].set_yscale('log')
    plt.tight_layout()

    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
        for text in axi.get_legend().get_texts(): text.set_fontsize(font_size)
    fig.set_tight_layout(True)
    pl.dump(fig, open("Sel_figs/plot_OSPA-swidth-Nsens.pickle", "wb"))
    fig.savefig('Sel_figs/plot_OSPA-swidth-Nsens.pdf')
    #%%