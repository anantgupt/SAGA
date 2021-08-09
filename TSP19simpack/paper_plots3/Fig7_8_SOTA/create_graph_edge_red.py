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
    Navg = 4
    Num_CPU = 4
    pmiss_str='0_05'#'0','0_4','0_05','0_2'
    fig_handle = []
    lbl = ['All edges','Pruned edges','F(A) evals','L(A) evals']
    pnos = [11]
    for pno in pnos:
        fig_handle1 = pl.load(open('SAGA/fig_Nob-pmiss'+pmiss_str+'/plot'+str(pno)+'.pickle','rb'))
        
    Ns = 1 # 0 for 4, 1 for 8
    Nss = ['4','8']
    Nsens = 6
    l_idx = [2,3]
    mkr = ['o','x','^','v','*']
    lst = ['-','-','-','-']
    colr=['r','b','g','g']
    fig, ax = plt.subplots(1,1)

    
    rng = fig_handle1.axes[1].lines[2].get_data()[0]
    for num in range(4): # SOTA loop
        rtm = fig_handle1.axes[1].lines[num].get_data()[1]
        # if num==2: % F(A) evaluations
            # ax.plot(rng, mse1/Navg, color=colr[num], linestyle=lst[1], marker=mkr[num])
        ax.plot(rng, rtm*Num_CPU, color=colr[num], linestyle=lst[num], marker=mkr[num], label=lbl[num])# label=r'$\mathcal{L}(\mathbf{t})$ Ops. '+SOTA[num])
    ax.plot(rng, rng*Nsens,'k:',label='Lower bound')
    ax.plot(rng, [sum([float(r+1)**i for i in range(Nsens+1)]) for r in rng],'k--',label='Upper bound')
    ax.legend(loc='best'),ax.grid(True);
    # ax.set_title('Algorithm Comparison ');
    ax.set_ylabel(r'No. of edges/evaluations')
   
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
    fig.savefig('Graph_edges.pdf')
    