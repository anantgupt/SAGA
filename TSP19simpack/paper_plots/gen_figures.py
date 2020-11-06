# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 20:54:50 2020
Generates plots in respective folders Fig3 - Fig12
1. Copy Nob-snr results for NOMP from OSPA directory to Fig_9_DFT directory (to avoid repeated simulations).
2. Copy and run create_fig... scripts to respective folders to generate the plots.

@author: gupta
"""
import numpy as np
from datetime import date, datetime
import argparse, os
from glob import glob
from shutil import copyfile

def main():
    # Copy data 
    d_list = glob('Fig4_OSPA/SAGA/*/*')
    # print(d_list)
    for d in range(len(d_list)):
        temp = d_list[d].split('\\')
        if os.path.isdir('Fig9_DFT'+'/'+temp[1])==False:
            os.mkdir('Fig9_DFT'+'/'+temp[1])
        copyfile(temp[0]+'/'+temp[1]+'/'+temp[2], 'Fig9_DFT'+'/'+temp[1]+'/'+temp[2])
    copyfile('plotting_scripts/Exp_miss_vs_nsens.pickle', 'Fig11_12_Nsens/Exp_miss_vs_nsens.pickle')
    py_list = glob('plotting_scripts/*.py')
    for f in range(len(py_list)):
        temp = py_list[f].split('\\')
        copyfile(temp[0]+'/'+temp[1], temp[1][:-3]+'/'+temp[1])
        os.chdir('./'+temp[1][:-3])
        os.system('python '+temp[1])
        os.chdir('..')
    
if __name__ == "__main__":
    __spec__ = None

    main()