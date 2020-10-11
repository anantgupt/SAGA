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
    'legend.fontsize': 8, # was 10
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
    fig_handle1 = pl.load(open('fig_snr-Nob1/plot2.pickle','rb'))
    # fig_handle2 = pl.load(open('fig_snr-Nob10/plot2.pickle','rb'))
    fig_handle3 = pl.load(open('fig_snr-Nob20/plot2.pickle','rb'))
    fig_handle4 = pl.load(open('fig_snr-Nob30/plot2.pickle','rb'))
    fig_handle4b = pl.load(open('fig_snr-Nob30/plot4.pickle','rb'))
    lbl = ['Range ','Doppler ']
    lbl2 = ['Position ','Velocity ']
    fig, ax = plt.subplots(2,1)
    for i in range(2):
        rng = fig_handle1.axes[i].lines[0].get_data()[0]
        crb1 = fig_handle1.axes[i].lines[1].get_data()[1]
        mse1 = fig_handle1.axes[i].lines[0].get_data()[1]
        #cnt1 = fig_handle1.axes[3]
        # crb2 = fig_handle2.axes[i].lines[1].get_data()[1]
        # mse2 = fig_handle2.axes[i].lines[0].get_data()[1]
        #cnt2 = fig_handle2.axes[3]
        crb3 = fig_handle3.axes[i].lines[1].get_data()[1]
        mse3 = fig_handle3.axes[i].lines[0].get_data()[1]
        crb4b = fig_handle4b.axes[i].lines[1].get_data()[1]
        mse4b = fig_handle4b.axes[i].lines[0].get_data()[1]
        
        crb4 = fig_handle4.axes[i].lines[1].get_data()[1]
        mse4 = fig_handle4.axes[i].lines[0].get_data()[1]
        # ax[i].plot(rng, crb1, 'b--',label='CRB Nob=1')
        # ax[i].plot(rng, crb2, 'g--',label='CRB Nob=11')
        # ax[i].plot(rng, crb3, 'r--',label='CRB Nob=21')
        # ax[i].plot(rng, crb4, 'y--',label='CRB Nob=31')
        # ax[i].plot(rng, crb1, 'b--')
        # ax[i].plot(rng, crb2, 'g--')
        ax[i].plot(rng, crb4b, 'r--', label=lbl[i]+'CRB')
        ax[i].plot(rng, crb4, 'b--', label=lbl2[i]+'CRB')   
        # ax[i].plot(rng, mse1, 'b-', marker =',', label='Nob=1')
        # ax[i].plot(rng, mse2, 'g-',marker ='.', label='Nob=10')
        ax[i].plot(rng, mse4b, 'r-', marker ='+',label=lbl[i]+'RMSE')
        ax[i].plot(rng, mse4, 'b-', marker ='x',label=lbl2[i]+'RMSE')
        ax[i].legend(loc='upper right'),ax[i].grid(True);#ax[i].set_xlabel('SNR (dB)');
        ax[i].set_xlim(-20,5)
    ax[0].set_ylim(-31,11)
    ax[1].set_ylim(-28,11)
    ax[0].set_title('Range, Position Error');ax[0].set_ylabel('RMSE (dB) (m)')
    ax[1].set_title('Doppler, Velocity Error');ax[1].set_ylabel('RMSE (dB) (m/s)')
    ax[1].set_xlabel('SNR (dB)');
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    #plt.yscale('log')
    if height:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        height = 1.6*width*golden_mean # height in inches
        
    fig.set_size_inches(width, height, forward=True)
    
#    mpl.rcParams.update(params)

    # v--- change title and axeslabel font sizes manually
    for axi in fig.axes:
        for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] +
                     axi.get_xticklabels() + axi.get_yticklabels()):
            item.set_fontsize(font_size)
        for text in axi.get_legend().get_texts(): text.set_fontsize(font_size)
    fig.set_tight_layout(True)
    pl.dump(fig, open("plot_PVRD_error.pickle", "wb"))
    fig.savefig('plot_PVRD_error.pdf')