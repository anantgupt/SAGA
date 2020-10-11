#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:57:27 2019
Use this code to edit figures saved using pickle dump
@author: anantgupta
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pl
import numpy as np

params = {
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
#    'text.fontsize': 8, # was 10
    'legend.fontsize': 7, # was 10
    'xtick.labelsize': 8,
    'ytick.labelsize': 8, # 'backend': 'ps',
    'text.usetex': True,# 'text.latex.preamble': ['\\usepackage{gensymb}'],
    'font.size': 8,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'grid.linestyle':':',       # dotted
    'grid.linewidth':0.4,     # in points
    'lines.linewidth':0.5,
    'lines.markersize' : 4,
    'lines.markeredgewidth':0.4
}
mpl.rcParams.update(params)

# Load figure from disk and display
if True:
    mode = 'Relax'
    width = 3.45
    height = 2.6
    font_size = 8
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1b = pl.load(open('mle/fig_Nob-snr-10/plot3.pickle','rb'))
    fig_handle2b = pl.load(open('mle/fig_Nob-snr-15/plot3.pickle','rb'))
    fig_handle3b = pl.load(open('mle/fig_Nob-snr0/plot3.pickle','rb'))
    
    fig_handle1 = pl.load(open('relax/fig_Nob-snr-10/plot3.pickle','rb'))
    fig_handle2 = pl.load(open('relax/fig_Nob-snr-15/plot3.pickle','rb'))
    fig_handle3 = pl.load(open('relax/fig_Nob-snr0/plot3.pickle','rb'))
    
    fig, ax = plt.subplots(3,1)
    order = [2,0,1]
    for i in range(3):
        rng = fig_handle1.axes[i].lines[0].get_data()[0]
        crb1 = fig_handle1b.axes[i].lines[0].get_data()[1]
        mse1 = fig_handle1.axes[i].lines[0].get_data()[1]
        #cnt1 = fig_handle1.axes[3]
        crb2 = fig_handle2b.axes[i].lines[0].get_data()[1]
        mse2 = fig_handle2.axes[i].lines[0].get_data()[1]
    #    #cnt2 = fig_handle2.axes[3]
        crb3 = fig_handle3b.axes[i].lines[0].get_data()[1]
        mse3 = fig_handle3.axes[i].lines[0].get_data()[1]
        #cnt3 = fig_handle3.axes[3]
    #for i in range(3):
    #    for line in fig_handle2.axes[i].lines[:10]:
    #        mser[t]=line.get_data()[1]
    #        t+=1
        # if i ==2:
        #     crb3 = -crb3, crb2 = -crb2, crb1 = -crb1
        #     mse3 = -mse3, mse2=-mse2, mse1 = -mse1
        ax[order[i]].plot(rng, crb3, 'r--')
        ax[order[i]].plot(rng, crb1, 'b.--')
        ax[order[i]].plot(rng, crb2, 'g--+')
        
        ax[order[i]].plot(rng, mse3, 'r-', label='SNR=0 dB')
        ax[order[i]].plot(rng, mse1, 'b.-', label='SNR=-10 dB')
        ax[order[i]].plot(rng, mse2, 'g-+', label='SNR=-15 dB')
        
        ax[order[i]].legend(loc='best'),ax[order[i]].grid(True);
    #plt.plot(rng, cnt1, 'b:', label='Count SNR=-10 dB')
    #plt.plot(rng, cnt2, 'g:', label='Count SNR=0 dB')
    #plt.plot(rng, cnt3, 'r:', label='Count SNR=10 dB')
    #plt.errorbar(rng, np.sqrt(np.mean(mser**2, axis=0)), np.std(mser, axis=0), label='Range RMSE'),plt.yscale('log')
    ax[2].set_title('OSPA');ax[2].set_ylabel('OSPA (m)')
    ax[0].set_title('Localization Error');ax[0].set_ylabel('RMSE (m)')
    ax[1].set_title('Cardinality Error');ax[1].set_ylabel(r'$N_e-N_T$')
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    ax[2].set_yscale('log');ax[0].set_yscale('linear');ax[1].set_yscale('symlog');
    ax[2].set_xlabel(r'Num Targets, $N_T$');
    ax[2].set_ylim(0.01,2)
    ax[0].set_ylim(0,0.25)
#    ax[2].set_ylim(-31,-4)
#    ax[0].set_ylim(0.1,10);
#    ax[1].set_ylim(0.1,10);
    if height:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        height = 2.3*width*golden_mean # height in inches
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for axi in ax:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
    plt.tight_layout()

    pl.dump(plt.figure(1), open("plot_OSPA_error_Nob.pickle", "wb"))
    fig.savefig('plot_OSPA_error_Nob.pdf', Transparent=True)
    #%%
    #plt.figure(2)
    #fig_handle = pl.load(open('res_comb/plot_doppler_resolution.pickle','rb'))