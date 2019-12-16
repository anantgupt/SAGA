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
def cf4dft(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    #fig_handle1 = pl.load(open('fig_Nsens-Nob1/plot11.pickle','rb'))
    fig_handle2 = pl.load(open('fig_Nsens-Nob10/plot11.pickle','rb'))
    fig_handle4 = pl.load(open('fig_Nsens-Nob20/plot11.pickle','rb'))

    fig_handle2d = pl.load(open('fig_DFT_Nsens-Nob10/plot11.pickle','rb'))
    fig_handle4d = pl.load(open('fig_DFT_Nsens-Nob20/plot11.pickle','rb'))

    fig_handle2b = pl.load(open('fig_Nsens-Nob10/plot1.pickle','rb'))
    fig_handle4b = pl.load(open('fig_Nsens-Nob20/plot1.pickle','rb'))
    fig_handle2bd = pl.load(open('fig_DFT_Nsens-Nob10/plot1.pickle','rb'))
    fig_handle4bd = pl.load(open('fig_DFT_Nsens-Nob20/plot1.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, axa = plt.subplots(1,2)
    lbl=[mode+" All edges",mode+" Brute",mode+'-Edges',mode+'-LLR']
    mkr = [',','.','+','x','*']
    ax = axa[0]
    for i in range(4):
        rng = fig_handle2.axes[1].lines[i].get_data()[0]

        mse2 = fig_handle2.axes[1].lines[i].get_data()[1]

        mse4 = fig_handle4.axes[1].lines[i].get_data()[1]
        mse2d = fig_handle2d.axes[1].lines[i].get_data()[1]

        mse4d = fig_handle4d.axes[1].lines[i].get_data()[1]

        ax.plot(rng, mse2, 'g-'+mkr[i], label=lbl[i]+', NOMP Nob=10')
        ax.plot(rng, mse4, 'r-'+mkr[i], label=lbl[i]+', NOMP Nob=20')
        ax.plot(rng, mse2, 'b--'+mkr[i], label=lbl[i]+', DFT Nob=10')
        ax.plot(rng, mse4, 'y--'+mkr[i], label=lbl[i]+', DFT Nob=20')
    
    r2 = fig_handle2b.axes[1].lines
    r4 = fig_handle4b.axes[1].lines
    r2d = fig_handle2bd.axes[1].lines
    r4d = fig_handle4bd.axes[1].lines
    lbl2=['Estimation','',mode+'-Assocaition','']
    idxa = [0,2]
    for i in idxa:
        axa[1].plot(rng, r2[i].get_data()[1], 'g-'+mkr[i], label='NOMP Nob=10 '+lbl2[i])
        axa[1].plot(rng, r4[i].get_data()[1], 'r-'+mkr[i], label='NOMP Nob=20'+lbl2[i])
        axa[1].plot(rng, r2d[i].get_data()[1], 'b--'+mkr[i+1], label='DFT Nob=10 '+lbl2[i])
        axa[1].plot(rng, r4d[i].get_data()[1], 'y--'+mkr[i+1], label='DFT Nob=20'+lbl2[i])
    
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('Num Sensors');
    ax.set_title('Association Complexity');ax.set_ylabel('Number of tracks visited')
    #ax[1].set_title('Localization Error');ax[1].set_ylabel('RMSE (dB)')
    #ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
    plt.tight_layout()
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    ax.set_yscale('log')
    axa[1].set_yscale('log')
    axa[1].grid(True)
    axa[1].legend(loc='best')
    
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.tight_layout()
    pl.dump(fig, open("Sel_figs/DFT_plot_complexity_vs_Nsens-Nob.pickle", "wb"))
    fig.savefig('Sel_figs/DFT_plot_complexity_vs_Nsens-Nob.pdf')
    #%%
    #plt.figure(2)
    #fig_handle = pl.load(open('res_comb/plot_doppler_resolution.pickle','rb'))