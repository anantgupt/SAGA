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
def cf4d(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_Nob-Nsens4/plot11.pickle','rb'))
    fig_handle2 = pl.load(open('fig_Nob-Nsens6/plot11.pickle','rb'))
    fig_handle3 = pl.load(open('fig_Nob-Nsens4/plot1.pickle','rb'))
    fig_handle4 = pl.load(open('fig_Nob-Nsens6/plot1.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, axa = plt.subplots(1,2)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    lbl=[mode+" All edges",mode+" Brute",mode+'-Edges',mode+'-LLR']
    mkr = [',','.','+','x','*']
    ax = axa[0]
    for i in range(4):
        rng = fig_handle2.axes[1].lines[i].get_data()[0]
    #    crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        mse1[i] = fig_handle1.axes[1].lines[i].get_data()[1]
        #cnt1 = fig_handle1.axes[3]
    #    crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        mse2[i] = fig_handle2.axes[1].lines[i].get_data()[1]
        ax.plot(rng, mse1[i], 'g-'+mkr[i], label=lbl[i]+r', $N_{s}=4$')
#    ax.plot(rng, mse2[0], 'b--', label='Brute, Nob=10')
        ax.plot(rng, mse2[i], 'r-'+mkr[i], label=lbl[i]+r', $N_{s}=6$')
        
    r2 = fig_handle3.axes[1].lines
    r4 = fig_handle4.axes[1].lines
    lbl2=['Estimation','',mode+'-Assocaition','']
    idxa = [0,2]
    for i in idxa:
        axa[1].plot(rng, r2[i].get_data()[1], 'g-'+mkr[i], label=r'$N_s=4$ '+lbl2[i])
        axa[1].plot(rng, r4[i].get_data()[1], 'r-'+mkr[i], label=r'$N_s=6$'+lbl2[i])
    
    #ax.plot(rng, mse4[0], 'r--s', label='All edges, Rob=2')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('Num targets');
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
    
    fig.set_tight_layout(True)
    pl.dump(fig, open("Sel_figs/plot_complexity_vs_Nob-Nsens.pickle", "wb"))
    fig.savefig('Sel_figs/plot_complexity_vs_Nob-Nsens.pdf')