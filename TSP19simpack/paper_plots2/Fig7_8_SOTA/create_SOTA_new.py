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
import matplotlib.ticker as ticker

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
    golden_mean = (np.sqrt(5)-1.0)/2.0
    height = width*golden_mean
    font_size = 8
    Navg = 100
    pmiss_str='0_05'#'0','0_4','0_05','0_2'
    fig_handle = []
    SOTA = ['MCF','SAESL','SAGA','NN']
    pnos = [1,3,11]
    for pno in pnos:
        fig_handle1 = pl.load(open('MCF/fig_Nob-pmiss'+pmiss_str+'/plot'+str(pno)+'.pickle','rb'))
        fig_handle2 = pl.load(open('SAESL/fig_Nob-pmiss'+pmiss_str+'/plot'+str(pno)+'.pickle','rb'))

        fig_handle1d = pl.load(open('SAGA/fig_Nob-pmiss'+pmiss_str+'/plot'+str(pno)+'.pickle','rb'))
        fig_handle2d = pl.load(open('NN/fig_Nob-pmiss'+pmiss_str+'/plot'+str(pno)+'.pickle','rb'))

        fig_handle.append([fig_handle1,fig_handle2,fig_handle1d,fig_handle2d])
    Ns = 1 # 0 for 4, 1 for 8
    Nss = ['4','8']
    l_idx = [2,3]
    mkr = [',','.','+','x','*']
    lst = ['-',':','--','-.']
    colr=['r','b','g','m']
    fig, ax = plt.subplots(1,1)
    # axa = ax.twinx()
    fig2, axa = plt.subplots(1,1)
    # axa2=axa.twinx()
    fig3, axaa = plt.subplots(1,1)
    # axaa2=axaa.twinx()
    fig4, axaaa = plt.subplots(1,1)

    for num in range(4): # SOTA loop
        rng = fig_handle[2][num].axes[1].lines[2].get_data()[0]
        mse1 = fig_handle[2][num].axes[1].lines[2].get_data()[1]
        mse2 = fig_handle[2][num].axes[1].lines[3].get_data()[1]
        ospa = fig_handle[1][num].axes[0].lines[0].get_data()[1]
        card = fig_handle[1][num].axes[2].lines[0].get_data()[1]
        rtm = fig_handle[0][num].axes[1].lines[2].get_data()[1]
        # if num==2: % F(A) evaluations
            # ax.plot(rng, mse1/Navg, color=colr[num], linestyle=lst[1], marker=mkr[num])
        ax.plot(rng, mse2/Navg, color=colr[num], linestyle=lst[0], marker=mkr[num], label=SOTA[num])# label=r'$\mathcal{L}(\mathbf{t})$ Ops. '+SOTA[num])
        axa.plot(rng, ospa, color=colr[num], linestyle=lst[0], marker=mkr[num], label=SOTA[num])
        if num==0:
            axaa.plot(rng, 0*np.ones(len(rng)),'k:',linewidth = 2, label='true')
        if True: #num>0 and num<3:
        	axaa.plot(rng, -card, color=colr[num], linestyle=lst[1], marker=mkr[num], label=SOTA[num])
        # axa2.plot(rng, rtm, color=colr[num], linestyle=lst[2], marker=mkr[num], label=r'Runtime, '+SOTA[num])
        axaaa.plot(rng, rtm/Navg, color=colr[num], linestyle=lst[0], marker=mkr[num], label=SOTA[num])
        
    ax.legend(loc='best'),ax.grid(True);
    # ax.set_title('Algorithm Comparison ');
    ax.set_ylabel(r'Likelihood Evaluations')
   
    ax.set_yscale('log');
    ax.set_xlabel('Num Targets');
    # ax.set_ylim(0.01,10)
    # axa.set_ylim(0,0.3)

    fig.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)
    fig.set_tight_layout(True)

    # pl.dump(fig, open("SOTA_Complex_comparison_vs_Nob-Nsens"+Nss[Ns]+".pickle", "wb"))
    fig.savefig('SOTA_Complex_comparison_vs_Nob-Nsens'+Nss[Ns]+'.pdf')
    #######
    axa.legend(loc='best'),axa.grid(True);axa.set_yscale('log')
    # axa.set_title('Algorithm Comparison ');#axa2.set_ylabel('RunTime (s)')
    axa.set_xlabel('Num Targets');axa.set_ylabel('OSPA')
    fig2.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([axa.title, axa.xaxis.label, axa.yaxis.label] +
                 axa.get_xticklabels() + axa.get_yticklabels()):
        item.set_fontsize(font_size)
    fig2.set_tight_layout(True)
    # pl.dump(fig2, open("SOTA_RTIME_vs_OSPA_comparison_vs_Nob"+Nss[Ns]+".pickle", "wb"))
    fig2.savefig('SOTA_RTIME_vs_OSPA_comparison_vs_Nob'+Nss[Ns]+'.pdf')
    ##############
    axaa.legend(loc='best'),axaa.grid(True);axaa.set_yscale('linear')
    axaa.set_title('Number of targets missed');#axaa.set_ylim([-1,14])
    axaa.set_xlabel('Num Targets detected');axaa.set_ylabel('Missed Targets')
    axaa.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig3.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([axaa.title, axaa.xaxis.label, axaa.yaxis.label] +
                 axaa.get_xticklabels() + axaa.get_yticklabels()):
        item.set_fontsize(font_size)
    fig3.set_tight_layout(True)
    # pl.dump(fig3, open("SOTA_RTIME_vs_Card_comparison_vs_Nob"+Nss[Ns]+".pickle", "wb"))
    fig3.savefig('SOTA_RTIME_vs_Card_comparison_vs_Nob'+Nss[Ns]+'.pdf')
    ##############
    axaaa.legend(loc='best'),axaaa.grid(True);axaaa.set_yscale('log')
    # axaaa.set_title('Algorithm Comparison ');#axa2.set_ylabel('RunTime (s)')
    axaaa.set_xlabel('Num Targets');axaaa.set_ylabel('Runtime (s)')
    fig4.set_size_inches(width, height, forward=True)
    # v--- change title and axeslabel font sizes manually
    for item in ([axaaa.title, axaaa.xaxis.label, axaaa.yaxis.label] +
                 axaaa.get_xticklabels() + axaaa.get_yticklabels()):
        item.set_fontsize(font_size)
    fig4.set_tight_layout(True)
    # pl.dump(fig2, open("SOTA_RTIME_comparison_vs_Nob"+Nss[Ns]+".pickle", "wb"))
    fig4.savefig('SOTA_RTIME_comparison_vs_Nob'+Nss[Ns]+'.pdf')