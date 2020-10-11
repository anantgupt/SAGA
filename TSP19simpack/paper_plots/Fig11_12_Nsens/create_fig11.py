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
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    Navg = 100
    fha = [[0 for _ in range(4)] for _ in range(5)]
    si=['0','1','2','4','7']
    for i in range(5):
        fha[i][0] = pl.load(open('relax/fig_Nsens-rob'+si[i]+'/plot3.pickle','rb'))
        fha[i][1] = pl.load(open('relax/fig_Nsens-rob'+si[i]+'/plot1.pickle','rb')) # Use plot11 for ops
        fha[i][2] = pl.load(open('mle/fig_Nsens-rob'+si[i]+'/plot3.pickle','rb'))
        fha[i][3] = pl.load(open('mle/fig_Nsens-rob'+si[i]+'/plot1.pickle','rb')) # Use plot11 for ops

    ax=[0]*2
    lbl=[r'$\rho=0$', r'$\rho=1$',r'$\rho=2$',r'$\rho=4$',r'$\rho=7$', r'$\rho=7$',r'$\rho=\infty$',r'SAESL']
    colr = ['r','b','g','m']
    mkr = [',','.','+','x','^','+','d']
    ls = ['-','--',':']
    fig, ax = plt.subplots(1,1)
    ax2 = ax.twinx()
    idxa= [0,1,2,3,4]
    for i in idxa:
        for j in range(2):
            fh1 = fha[i][j]
            rng = fh1.axes[1].lines[0].get_data()[0]
            
            if j==0:
                mse = fh1.axes[0].lines[0].get_data()[1]
                ax.plot(rng, mse, colr[1]+mkr[i]+ls[j],label = lbl[i+4*j])
            else:
                mse = fh1.axes[1].lines[2].get_data()[1]+fh1.axes[1].lines[3].get_data()[1]
                # mse =  fh1.axes[1].lines[2].get_data()[1]
                ax2.plot(rng, mse/Navg, colr[0]+mkr[i]+ls[j])
        # print(len(fha[1][1].lines))
    
    rng = fha[-1][2].axes[0].lines[0].get_data()[0]
    ax.plot(rng, fha[-1][2].axes[0].lines[0].get_data()[1], colr[1]+ls[2], linewidth=1, label = lbl[-1])
    ax2.plot(rng, (fha[-1][3].axes[1].lines[2].get_data()[1]+fha[-1][3].axes[1].lines[3].get_data()[1])/Navg, colr[0]+ls[2], linewidth=1)    

    ax.legend(loc='upper left')
    ax.grid(True);
    # ax[i].set_title(fig_handle1.axes[2*i].get_title())
    ax.set_ylabel('OSPA')
    ax.yaxis.label.set_color(colr[1])


    ax2.set_ylabel('Runtime (s)')
    ax2.yaxis.label.set_color(colr[0])
    # ax2.set_ylim(1e5,1e6)
    # ax.set_ylim(0.15,0.6) 
    ax.set_xlabel('Num Sensors')
#    for axis in [ax[0].xaxis, ax[0].yaxis]:
#        formatter = ScalarFormatter()
#        formatter.set_scientific(False)
#        axis.set_major_formatter(formatter)
    ax2.set_yscale('log')


    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
        
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0g'))
    # ax2.tick_params(axis='y', which='minor', bottom=False)

#    ax.yaxis.set_minor_formatter(None)
    # plt.yticks([0.2,0.3,0.4,0.6,1])
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    fig.set_tight_layout(True)
    pl.dump(fig, open("plot_amb_vs_Nsens-rob.pickle", "wb"))
    fig.savefig('plot_amb_vs_Nsens-rob.pdf')