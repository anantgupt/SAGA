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
from matplotlib.ticker import FormatStrFormatter

params = {
    'axes.labelsize': 8, # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
#    'text.fontsize': 8, # was 10
    'legend.fontsize': 6, # was 10
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
    width = 3.45
    golden_mean = (np.sqrt(5)-1.0)/2.0
    height = width*golden_mean
    font_size = 8
    Navg = 50
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
    
    fig_handle4c = pl.load(open('fig_Nsens-Nob20/plot3.pickle','rb'))
    fig_handle4cd = pl.load(open('fig_DFT_Nsens-Nob20/plot3.pickle','rb'))

    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    fig, ax = plt.subplots(1,1)
    figa, axa = plt.subplots(1,1)
    lbl=[" All edges"," Brute",r'$\mathcal{F}(\mathbf{t})$',r'$\mathcal{L}(\mathbf{t})$']
    mkr = ['.','.','x','+','*']
    lst = ['--',':','-','-.']
    idx_rng = [2,3]
    for i in idx_rng:
        rng = fig_handle2.axes[1].lines[i].get_data()[0]
        rngb = fig_handle2d.axes[1].lines[i].get_data()[0]
        mse2 = fig_handle2.axes[1].lines[i].get_data()[1]

        mse4 = fig_handle4.axes[1].lines[i].get_data()[1]
        mse2d = fig_handle2d.axes[1].lines[i].get_data()[1]

        mse4d = fig_handle4d.axes[1].lines[i].get_data()[1]

        # ax.plot(rng, mse2, 'g-'+mkr[i], label=lbl[i]+', NOMP Nsens=4')
        ax.plot(rng, mse4, 'r'+lst[i]+mkr[0], label=lbl[i]+', NOMP')
        # ax.plot(rng, mse2, 'b--'+mkr[i], label=lbl[i]+', DFT Nsens=4')
        ax.plot(rng, mse4, 'b'+lst[i]+mkr[2], label=lbl[i]+', DFT')
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('Num Sensors');
    ax.set_title('Association Complexity');ax.set_ylabel('Number Operations')
    fig.set_size_inches(width, height, forward=True)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.tight_layout()
    pl.dump(fig, open("DFT_plot_OPS_complexity_vs_Nob-Nsens.pickle", "wb"))
    fig.savefig('DFT_plot_OPS_complexity_vs_Nob-Nsens.pdf')

    r2 = fig_handle2b.axes[1].lines
    r4 = fig_handle4b.axes[1].lines
    r2d = fig_handle2bd.axes[1].lines
    r4d = fig_handle4bd.axes[1].lines
    lbl2=['Estimation','','Association','']
    idxa = [0,2]
    colr= ['r','g','b']
    for i in idxa:
        # axa[1].plot(rng, r2[i].get_data()[1], 'g-'+mkr[i], label='NOMP Nsens=4 '+lbl2[i])
        axa.plot(rng, r4[i].get_data()[1]/Navg, colr[i],linestyle=lst[i],marker=mkr[0], label=lbl2[i]+', NOMP')
        # axa[1].plot(rngb, r2d[i].get_data()[1], 'b--'+mkr[i+1], label='DFT Nsens=4 '+lbl2[i])
        axa.plot(rngb, r4d[i].get_data()[1]/Navg, colr[i],linestyle=lst[i],marker=mkr[2], label=lbl2[i]+', DFT')
    # axa2 = axa.twinx()
    r4c = fig_handle4c.axes[1].lines
    r4cd = fig_handle4cd.axes[1].lines
    # axa2.plot(rng, r4c[0].get_data()[1], 'r'+lst[1]+mkr[0], label='NOMP',linewidth=1.2)
    # axa2.plot(rng, r4cd[0].get_data()[1], 'b'+lst[1]+mkr[2], label='DFT',linewidth=1.2)
    # axa.set_title('Association with DFT vs NOMP')
    # axa2.set_ylabel(r'\textbf{OSPA}')
    # axa2.set_ylim(0,0.5)
    
    axa.set_ylabel('Runtime (s)');
    plt.tight_layout()
    #plt.title('Doppler Resolution');plt.xlabel('Doppler Separation (m/s)');plt.ylabel('RMSE (dB), Count')
    
    axa.set_yscale('log')
    axa.set_xlabel('Num Sensors')
    axa.grid(True)
    axa.legend(loc='upper left')
    
    figa.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([axa.title, axa.xaxis.label, axa.yaxis.label] +
                 axa.get_xticklabels() + axa.get_yticklabels()):
        item.set_fontsize(font_size)
    plt.tight_layout()
    pl.dump(figa, open("DFT_plot_TIME_complexity_vs_Nob-Nsens.pickle", "wb"))
    figa.savefig('DFT_plot_TIME_complexity_vs_Nob-Nsens.pdf')
    #%%
    #plt.figure(2)
    #fig_handle = pl.load(open('res_comb/plot_doppler_resolution.pickle','rb'))