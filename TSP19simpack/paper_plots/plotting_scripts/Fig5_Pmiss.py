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
    'lines.markersize' : 2,
    'lines.markeredgewidth':0.4
}
mpl.rcParams.update(params)

# Load figure from disk and display
if True:
    width = 3.45
    golden_mean = (np.sqrt(5)-1.0)/2.0
    height = 1.6*width*golden_mean
    font_size = 8
    #fig_handle = pl.load(open('results6_14/fig_obj_est2/plot4.pickle','rb'))
    fig_handle1 = pl.load(open('SAGA/fig_pmiss-rob1/plot10.pickle','rb'))
    fig_handle2 = pl.load(open('SAGA/fig_pmiss-rob2/plot10.pickle','rb'))
    fig_handle3 = pl.load(open('SAGA/fig_pmiss-rob8/plot10.pickle','rb'))
    fig_handle4 = pl.load(open('SAESL/fig_pmiss-rob8/plot10.pickle','rb'))

    fig_handle1b = pl.load(open('SAGA/fig_pmiss-rob1/plot3.pickle','rb'))
    fig_handle2b = pl.load(open('SAGA/fig_pmiss-rob2/plot3.pickle','rb'))
    fig_handle3b = pl.load(open('SAGA/fig_pmiss-rob8/plot3.pickle','rb'))
    fig_handle4b = pl.load(open('SAESL/fig_pmiss-rob8/plot3.pickle','rb'))    
    fha = [fig_handle1, fig_handle2, fig_handle3,fig_handle4]
    fhab = [fig_handle1b, fig_handle2b, fig_handle3b,fig_handle4b]
    colr = ['r','b','g','m']
    mkr = [',','.','x','v']
    ls = ['-','--',':']
    #fig_handle3 = pl.load(open('fig_snr-Nob21/plot2.pickle','rb'))
    lbl=fig_handle1.axes[0].get_legend().get_texts()
    p2lbl = [r'SAGA, $\rho=1$',r'SAGA, $\rho=2$',r'SAGA, $\rho=4$',r'SAESL, $\rho=4$']
    fig, axa = plt.subplots(2,1)
    ax=axa[0]
    mse1=[0]*4;mse2=[0]*4;mse3=[0]*4;mse4=[0]*4;mse5=[0]*4
    for i in range(2):
        rng2 =  fha[i].axes[1].lines[0].get_data()[0]
        for j in range(4):
            rng = fha[2+i].axes[0].lines[2*j].get_data()[0]
            mse1 = fha[2+i].axes[0].lines[2*j].get_data()[1]
            if i==0:
                ax.plot(rng, mse1,colr[j]+mkr[i]+ls[i], label = r'$P_{miss}=$'+'{:0.2f}'.format(rng2[2*j]))
            else:
                ax.plot(rng, mse1,colr[j]+mkr[i]+ls[i])
    ax.legend(loc='best')
    ax.grid(True);
    
    # ax[i].set_title(fig_handle1.axes[2*i].get_title())
    ax.set_ylabel(fig_handle1.axes[0].get_ylabel())
    ax.set_xlabel(fig_handle1.axes[0].get_xlabel())
    ax.set_title(r'Graph size v/s iterations')
    
    # for i in range(4):    
    #     rng2 =  fhab[i].axes[2].lines[0].get_data()[0]
    #     mse2 =  fhab[i].axes[2].lines[0].get_data()[1]
    #     axa[1].plot(rng2, mse2, marker=mkr[i], linestyle=ls[i==2], label=p2lbl[i])
    
    # # axa[1].plot(rng2, 20-fha[i].axes[1].lines[1].get_data()[1], 'k:',label='true')

    # axa[1].set_ylabel(r'$N_T-N_e$')
    # axa[1].set_xlabel(r'$P_{miss}$')        
    # axa[1].legend(loc='best')
    # axa[1].grid(True);
    
    # axa[1].set_title(r'Cardinality Error')
    ax3=1
    for i in range(4):    
        rng2 =  fhab[i].axes[0].lines[0].get_data()[0]
        mse2 =  fhab[i].axes[0].lines[0].get_data()[1]
        axa[ax3].plot(rng2, mse2, marker=mkr[i], linestyle=ls[i==3], label=p2lbl[i])
    # axa[2].plot(rng2, fhab[i].axes[0].lines[0].get_data()[1], 'k:',label='true')
    axa[ax3].set_ylabel(r'OSPA')
    axa[ax3].set_xlabel(r'$P_{miss}$')        
    axa[ax3].legend(loc='best')
    axa[ax3].grid(True);
    axa[ax3].set_yscale('log')
    axa[ax3].set_title(r'OSPA')

    # ax[1].set_ylim(-4,12)
    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
        
#    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#    ax.yaxis.set_minor_formatter(None)
#    plt.yticks([0.2,0.3,0.4,0.6,1])
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    fig.set_tight_layout(True)
    pl.dump(fig, open("Graph_nodes_vs_pmiss.pickle", "wb"))
    fig.savefig('Graph_nodes_vs_pmiss.pdf')