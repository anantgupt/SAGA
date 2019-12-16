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
def cf4e(mode = 'Relax', width = 3.45, height = 2.6, font_size = 8):
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('fig_pmiss-rob0/plot11.pickle','rb'))
    fig_handle2 = pl.load(open('fig_pmiss-rob2/plot11.pickle','rb'))
    fig_handle3 = pl.load(open('fig_pmiss-rob4/plot11.pickle','rb'))
    fig_handle4 = pl.load(open('fig_pmiss-rob8/plot11.pickle','rb'))
    fig_handle1b = pl.load(open('fig_pmiss-rob0/plot1.pickle','rb'))
    fig_handle2b = pl.load(open('fig_pmiss-rob2/plot1.pickle','rb'))
    fig_handle3b = pl.load(open('fig_pmiss-rob4/plot1.pickle','rb'))
    fig_handle4b = pl.load(open('fig_pmiss-rob8/plot1.pickle','rb'))
    
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    
    fig, axa = plt.subplots(1,2)
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4
    lbl=[mode+" All edges",mode+" Brute",mode+'-Edges',mode+'-LLR']
    mkr = [',','.','+','x','*']
    ax=axa[0]
    for i in range(4):
        rng = fig_handle2.axes[1].lines[i].get_data()[0]
    #    crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        mse1[i] = fig_handle1.axes[1].lines[i].get_data()[1]
        mse2[i] = fig_handle2.axes[1].lines[i].get_data()[1]
        mse3[i] = fig_handle3.axes[1].lines[i].get_data()[1]
        mse4[i] = fig_handle4.axes[1].lines[i].get_data()[1]
        ax.plot(rng, mse1[i], 'g-'+mkr[i], label=lbl[i]+r', $\rho=0$')
        ax.plot(rng, mse2[i], 'b-'+mkr[i], label=lbl[i]+r', $\rho=2$')
        ax.plot(rng, mse3[i], 'r-'+mkr[i], label=lbl[i]+r', $\rho=4$')
        ax.plot(rng, mse4[i], 'm-'+mkr[i], label=lbl[i]+r', $\rho=8$')
        
    r1 = fig_handle1b.axes[1].lines
    r2 = fig_handle2b.axes[1].lines
    r3 = fig_handle3b.axes[1].lines
    r4 = fig_handle4b.axes[1].lines
    lbl2=['Estimation','',mode+'-Assocaition','']
    idxa = [0,2]
    for i in idxa:
        axa[1].plot(rng, r1[i].get_data()[1], 'g-'+mkr[i], label=r'$\rho=0$ '+lbl2[i])
        axa[1].plot(rng, r2[i].get_data()[1], 'r-'+mkr[i], label=r'$\rho=2$'+lbl2[i])
        axa[1].plot(rng, r3[i].get_data()[1], 'g-'+mkr[i], label=r'$\rho=4$ '+lbl2[i])
        axa[1].plot(rng, r4[i].get_data()[1], 'r-'+mkr[i], label=r'$\rho=8$'+lbl2[i])
    #ax.plot(rng, mse4[0], 'r--s', label='All edges, Rob=2')
    #    ax[i].plot(rng, mse3, 'r-', label='RMSE Nob=21')
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel(r'$P_{miss}$');
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
    pl.dump(fig, open("Sel_figs/plot_complexity_vs_pmiss-rob.pickle", "wb"))
    fig.savefig('Sel_figs/plot_complexity_vs_pmiss-rob.pdf')