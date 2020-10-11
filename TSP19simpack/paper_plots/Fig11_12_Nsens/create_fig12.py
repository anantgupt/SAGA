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

    fha = [[0 for _ in range(3)] for _ in range(5)]
    si=['0','1','2','4','7']
    fhtheo = pl.load(open('Exp_miss_vs_nsens.pickle','rb'))
    for i in range(4):
        fha[i][0] = pl.load(open('relax/fig_Nsens-rob'+si[i]+'/plot3.pickle','rb'))
        fha[i][1] = pl.load(open('relax/fig_Nsens-rob'+si[i]+'/plot11.pickle','rb'))
        fha[i][2] = pl.load(open('mle/fig_Nsens-rob'+si[i]+'/plot11.pickle','rb'))

    ax=[0]*2
    lbl=[r'$\rho=0$', r'$\rho=1$',r'$\rho=2$',r'$\rho=4$',r'Brute, $\rho=0$', r'Brute $\rho=1$',r'Brute $\rho=2$',r'Brute $\rho=4$']
    colr = ['r','b','g','m']
    mkr = [',','.','x','v']
    ls = ['-','--',':']
    fig, ax = plt.subplots(1,1)
    # 
    for i in range(4):
        for j in range(1):
            fh1 = fha[i][j]
            rng = fh1.axes[2].lines[0].get_data()[0]
            mse = fh1.axes[2].lines[0].get_data()[1]
            if j==0:
                ax.plot(rng, mse, colr[1]+mkr[i]+ls[j],label = lbl[i+4*j])
            else:
                ax.plot(rng, mse, colr[1]+mkr[i]+ls[j])
            rng2 = fhtheo.axes[0].lines[int(si[i])].get_data()[0]
            ax.plot(rng2, (fhtheo.axes[0].lines[int(si[i])].get_data()[1]), colr[2]+mkr[i]+ls[2])
            
    ax.legend(loc='best')
    ax.grid(True);
    ax.set_ylabel('Cardinality Error')
    # ax.set_ylim(-19,4) 
    ax.yaxis.label.set_color(colr[1])
    if 0:
        ax2 = ax.twinx()
        for i, sis in enumerate(si):
            iv = int(sis)
            rng = fhtheo.axes[0].lines[iv].get_data()[0]
            ax.plot(rng, (fhtheo.axes[0].lines[iv].get_data()[1]), colr[2]+mkr[i]+ls[2], label=r'theo, $\rho=$'+str(i))
            ax2.plot(rng, fha[i][1].axes[1].lines[2].get_data()[1]+fha[i][1].axes[1].lines[3].get_data()[1],colr[0]+mkr[i]+ls[1])
            ax2.plot(rng, fha[i][2].axes[1].lines[2].get_data()[1]+fha[i][2].axes[1].lines[3].get_data()[1],'m-.')    # Rntime of MLE
        ax2.set_ylabel(r'Runtime (ops)')
        ax2.yaxis.label.set_color(colr[0])
        # ax2.set_ylim(0.5e2,1.5e4)
    ax.set_xlabel('Num Sensors')
#    for axis in [ax[0].xaxis, ax[0].yaxis]:
#        formatter = ScalarFormatter()
#        formatter.set_scientific(False)
#        axis.set_major_formatter(formatter)
	

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
    pl.dump(fig, open("plot_missed_vs_Nsens-rob.pickle", "wb"))
    fig.savefig('plot_missed_vs_Nsens-rob.pdf')