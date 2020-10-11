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
    Navg = 100
    fig_handle1 = pl.load(open('fig_DFT_Nob-snr-10/plot4.pickle','rb'))
    fig_handle2 = pl.load(open('fig_DFT_Nob-snr-15/plot4.pickle','rb'))
    fig_handle3 = pl.load(open('fig_Nob-snr-10/plot4.pickle','rb'))
    fig_handle4 = pl.load(open('fig_Nob-snr-15/plot4.pickle','rb'))

    fig_handle5 = pl.load(open('fig_DFT_Nob-snr-15/plot6.pickle','rb'))
    fig_handle6 = pl.load(open('fig_Nob-snr-15/plot6.pickle','rb'))

    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    fig, ax = plt.subplots(1,1)
    lbl=[" All edges"," Brute",r'$\mathcal{F}(\mathbf{t})$',r'$\mathcal{L}(\mathbf{t})$']
    mkr = [',','.','x','+','*']
    lst = ['--',':','-','-.']
    idx_rng = [0]
    ax2 = ax.twinx()
    for i in idx_rng:
        rng = fig_handle2.axes[0].lines[i].get_data()[0]
        crb = fig_handle2.axes[0].lines[1].get_data()[1]

        mse2 = fig_handle2.axes[0].lines[i].get_data()[1]

        mse4 = fig_handle4.axes[0].lines[i].get_data()[1]

        card2 = np.mean([k.get_data()[1] for k in fig_handle5.axes[2].lines],0)
        # for k in fig_handle5.axes[2].lines:
        # 	print(k.get_data()[1] )
        card4 = np.mean([k.get_data()[1] for k in fig_handle6.axes[2].lines],0)

        ax.plot(rng, mse2, 'b-x', label='DFT')
        ax.plot(rng, mse4, 'b-',marker='.', label='NOMP')
        ax.plot(rng, crb, 'k--', label='CRB')

        ax2.plot(rng, -card2, 'g-x', label='DFT', linewidth=1.1)
        ax2.plot(rng, -card4, 'g-',marker='.', label='NOMP', linewidth=1.1)

    ax.yaxis.label.set_color('b')
    ax2.yaxis.label.set_color('g')
    ax.legend(loc='best'),ax.grid(True);ax.set_xlabel('Num Targets');
    ax2.set_ylabel('Cardinality error');
    ax.set_ylabel('RMSE (dB)')
    fig.set_size_inches(width, height, forward=True)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    fig.set_tight_layout(True)
    pl.dump(fig, open("DFT_plot_Mod.pickle", "wb"))
    fig.savefig('DFT_plot_Mod.pdf')
