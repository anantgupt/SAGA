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
    Navg = 50 # Monte carlo trials
    Fcost = [1, 10]
    Lcost = [10, 1]
    fha = [[0 for _ in range(4)] for _ in range(3)]
    si=['0','20']
    for i in range(2):
        for j in range(2):
            if j==0:
                fha[i][0] = pl.load(open('SAGA/fig_Nob-rob'+si[i]+'/plot1.pickle','rb'))
                fha[i][1] = pl.load(open('SAGA/fig_Nob-rob'+si[i]+'/plot11.pickle','rb'))
                
            else:
                fha[i][2] = pl.load(open('SAESL/fig_Nob-rob'+si[i]+'/plot1.pickle','rb'))
                fha[i][3] = pl.load(open('SAESL/fig_Nob-rob'+si[i]+'/plot11.pickle','rb'))
    # fig_handle1 = pl.load(open('SAGA/fig_Nob-rob0/plot11.pickle','rb'))
    # fig_handle2 = pl.load(open('SAGA/fig_Nob-rob1/plot11.pickle','rb'))
    # fig_handle3 = pl.load(open('SAGA/fig_Nob-rob20/plot11.pickle','rb'))
    # fig_handle1b = pl.load(open('SAGA/fig_Nob-rob0/plot1.pickle','rb'))
    # fig_handle2b = pl.load(open('SAGA/fig_Nob-rob1/plot1.pickle','rb'))
    # fig_handle3b = pl.load(open('SAGA/fig_Nob-rob20/plot1.pickle','rb'))

    # fig_handle1aa = pl.load(open('SAESL/fig_Nob-rob0/plot11.pickle','rb'))
    # fig_handle2aa = pl.load(open('SAESL/fig_Nob-rob1/plot11.pickle','rb'))
    # fig_handle3aa = pl.load(open('SAESL/fig_Nob-rob20/plot11.pickle','rb'))
    # fig_handle1bb = pl.load(open('SAESL/fig_Nob-rob0/plot1.pickle','rb'))
    # fig_handle2bb = pl.load(open('SAESL/fig_Nob-rob1/plot1.pickle','rb'))
    # fig_handle4bb = pl.load(open('SAESL/fig_Nob-rob20/plot1.pickle','rb'))
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    ax=[0]*2
    lbl=[r'SAGA, $\rho=0$',r'SAGA $\rho=4$',r'SAESL, $\rho=0$', r'SAESL $\rho=4$']
    colr = ['r','b','g']
    mkr = [',','.','x','v']
    ls = ['-','--',':']
    fig, ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    for i in range(2):
        for j in range(4):
            fh1 = fha[i][j]
            rng = fh1.axes[1].lines[2].get_data()[0]
            if j%2==1:
                mse = (Fcost[j>1]*fh1.axes[1].lines[2].get_data()[1]+
                        Lcost[j>1]*fh1.axes[1].lines[3].get_data()[1])
                ax.semilogy(rng, mse, colr[1]+mkr[i]+ls[j>1],label = lbl[i+2*(j>1)])
            else:
                mse = fh1.axes[1].lines[2].get_data()[1]
                ax2.semilogy(rng, mse/Navg, colr[0]+mkr[i]+ls[j>1])
            
    ax.legend(loc='best')
    ax.grid(True);
    # ax[i].set_title(fig_handle1.axes[2*i].get_title())
    ax.set_ylabel('Operations')
    ax.yaxis.label.set_color(colr[1])
    ax2.set_ylabel('Runtime (s)')
    ax2.yaxis.label.set_color(colr[0])
    ax2.set_ylim(0.02,0.4e3)
    #ax[1].set_title('Localization Error');ax[1].set_ylabel('RMSE (dB)')
    #ax[2].set_title('Cardinality Error');ax[2].set_ylabel('Num Targets error')
   
    ax.set_xlabel('Num Targets')
#    for axis in [ax[0].xaxis, ax[0].yaxis]:
#        formatter = ScalarFormatter()
#        formatter.set_scientific(False)
#        axis.set_major_formatter(formatter)
#    ax.set_yscale('log')

    ax.set_ylim(2e2,4e6)
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
        
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax.yaxis.set_minor_formatter(None)
    # plt.yticks([0.2,0.3,0.4,0.6,1])
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    fig.set_tight_layout(True)
    pl.dump(fig, open("plot_complexity_vs_Nob-rob.pickle", "wb"))
    fig.savefig('plot_complexity_vs_Nob-rob.pdf')