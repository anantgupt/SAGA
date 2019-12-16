#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:04:25 2019
Methods to evaluate performance of localization algo
@author: anantgupta
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from GAutils import objects as ob
from GAutils import proc_est as pr

MISS_THRESHOLD = 1 # 1 for simulations (RD v/s SNR to get miss), 100 for Resolution
def plotg(res, sigmar, plt, fit=True):
    count, bins, ignored = plt.hist(res, density=True)
    if fit:
        plt.plot(bins, 1/(sigmar * np.sqrt(2 * np.pi)) *
        np.exp( - (bins)**2 / (2 * sigmar**2) ),
        linewidth=2, color='r')
    
def compute_ospa(true_scene0, est_scene, sensors, gardat=[], loc_wt=[1,1,1,1], p=2, c=MISS_THRESHOLD):
    # loc_wt: 4 D weights for x,y,vx,vy
    # c: Cardinality error weight
    # p: Localization error weight
    # If gardat available recompute true shifted scene
    if gardat:
        new_scene = []
        for (ri,rj,di,dj) in zip(gardat[0].r,gardat[1].r,gardat[0].d,gardat[1].d):
            new_scene.append(pr.get_pos_from_rd(ri, rj, di, dj, 0 , 1, sensors))
        true_scene = new_scene
    else:
        true_scene = true_scene0
    # get the minimal cost assignment
    cost = np.zeros([len(est_scene), len(true_scene)])
    for i, est_ob in enumerate(est_scene):
        for j, true_ob in enumerate(true_scene):
#            print('T=({},{}),E=({},{})'.format(true_ob.x,true_ob.y,est_ob.x,est_ob.y))
            cost[i,j],_,_ = compute_pos_error(est_ob, true_ob, loc_wt)
    row_ind, col_ind = linear_sum_assignment(cost) # Using hungarian method
    # compute the OSPA metric
    total_loc, ospa_loc = 0,0
    err_pos, err_vel =0,0
    c = 5*np.sqrt(min([np.sum(sensor.getnominalCRB()) for sensor in sensors])) # previous
    cpos = 10*np.sqrt(min([np.sum(sensor.getnominalCRB()[:2]) for sensor in sensors]))
    c_pen = 1 # Penalty for miss: 1
    n = max(len(est_scene), len(true_scene)) # Total est count
    m = min(len(est_scene), len(true_scene)) # Count for err_pos, err_vel
    ntrue = 0 # Total targets estimated close to truth
    PVerror = np.zeros((len(true_scene),2))
    for dn, (i,j) in enumerate(zip(row_ind, col_ind)):
        ct, cp, cv = compute_pos_error(est_scene[i], true_scene[j], loc_wt)
        if cp<cpos**2: # was ct<c: # Declare miss otherwise
            total_loc += ct**p
            ospa_loc += min(ct**p,c**p)
            err_pos += cp
            err_vel += cv
            PVerror[j,:]=[cp,cv]
            ntrue += 1 # Count number of actual targets estimates (CLose to truth)
        else:
            total_loc += 0
            ospa_loc += 0
            err_pos += (c_pen/2)**p
            err_vel += (c_pen/2)**p
            PVerror[j,:]=[cp, cv] #[(c/2)**p,(c/2)**p]
    if m==0:# If nothing good found, give out 1 target
        virt_trgt = ob.PointTarget(0, 0.1, 0, 0)
        for j in range(n):
            ct, cp, cv = compute_pos_error(virt_trgt, true_scene[j], loc_wt)
            total_loc += 0# ct**p
            ospa_loc += 0
            err_pos += cp
            err_vel += cv
            PVerror[j,:]=[cp,cv]
            m+=1
#    err_cn += len(est_scene)-len(true_scene) #(float(c**p*(n-m))/n)**(1/p)
    err_loc = (float(total_loc)/max(ntrue,1))**(1/p) 
    err_pos, err_vel = np.sqrt(float(err_pos)/m) , np.sqrt(float(err_vel)/m)
    ospa_err = ( float(ospa_loc + max(n-ntrue,0)* (c_pen**p)) / n)**(1/p) # If n>m count extra misses
    ospa_tuple = np.array([ospa_err,err_loc, len(est_scene)-len(true_scene), ntrue, err_pos]) 
    return ospa_tuple, PVerror
    
def compute_pos_error(ob1, ob2, loc_wt):
    xer = loc_wt[0] *abs(ob1.x - ob2.x)**2
    yer = loc_wt[1] *abs(ob1.y - ob2.y)**2
    vxer = loc_wt[2] *abs(ob1.vx - ob2.vx)**2
    vyer = loc_wt[3] *abs(ob1.vy - ob2.vy)**2
    return np.sqrt(xer+yer+vxer+vyer), xer+yer, vxer+vyer # distance
    
def compute_rd_error(garda_est, garda_true, plt=[]):
    r_mse = np.zeros(len(garda_est))
    d_mse = np.zeros(len(garda_est))
    Ns = len(garda_true)
    for si, (gard_est, gard_true) in enumerate(zip(garda_est, garda_true)):# mse for each sensor
        # NOTE: Adjust r,d before computing error to achieve CRB
        R_mat = np.subtract.outer(gard_est.r,gard_true.r)**2
        D_mat = np.subtract.outer(gard_est.d,gard_true.d)**2
        Np = min(len(gard_est.r), len(gard_true.r))
        if 1: # Use Hungarian method
            row_ind, col_ind = linear_sum_assignment(R_mat+D_mat)
            r_mse[si]=R_mat[row_ind, col_ind].sum()
            d_mse[si]=D_mat[row_ind, col_ind].sum()
        # else:
            # NOTE : Elementary minimum finding is not optimal!
            # for i in range(Np):
            #     [row, col] = unravel_index(np.argmin(R_mat), R_mat.shape)
            #     r_mse[si] += R_mat[row, col]
            #     d_mse[si] += D_mat[row, col]
            #     D_mat = np.delete(D_mat, row, axis=0); D_mat = np.delete(D_mat, col, axis=1)
            #     R_mat = np.delete(R_mat, row, axis=0); R_mat = np.delete(R_mat, col, axis=1)
        if plt:
            plt.subplot(np.floor(np.sqrt(Ns)),np.ceil(Ns/np.floor(np.sqrt(Ns))),si+1)
            plt.plot(gard_true.r, gard_true.d, 'k.')
            plt.plot(gard_est.r, gard_est.d, 'rx')
            for i in range(Np):
                plt.text(gard_est.r[row_ind[i]], gard_est.d[row_ind[i]], (col_ind[i])+1, FontSize=7)
            plt.title('Sensor {}'.format(si+1) )
            plt.grid(True)
    return r_mse, d_mse

def compute_rde_targetwise(garda_est, garda_true, sensors, plt=[],c=MISS_THRESHOLD):# RD error targetwise
    Ns = len(garda_true)
    Nt = len(garda_true[0].r)# All true gard have equal number of ons = N_target
    Nmiss = np.zeros(Ns)
    Nfa = np.zeros(Ns)
    dr = np.zeros((Ns, Nt))
    dd = np.zeros((Ns, Nt))
    crb = np.zeros((Ns, Nt, 2))# Expected based on gar_est gain
    present = np.zeros((Ns, Nt))
    for si, (gard_est, gard_true) in enumerate(zip(garda_est, garda_true)):# mse for each sensor
        # NOTE: Adjust r,d before computing error to achieve CRB
        R_mat = np.subtract.outer(gard_est.r,gard_true.r)
        D_mat = np.subtract.outer(gard_est.d,gard_true.d)
        Nobs = len(gard_est.r)
        if Nobs > Nt:
            Nfa[si] = Nobs - Nt
        else:
            Nmiss[si] = Nt - Nobs
        Np = min(Nobs, Nt)
        # Use Hungarian method
        cost = np.sqrt(R_mat**2+D_mat**2)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            if (cost[i, j]< c):
                present[si, j]=1
                dr[si, j]=R_mat[i, j]
                dd[si, j]=D_mat[i, j]
            else:
                Nmiss[si]+=1
                Nfa[si]+=1 # NOTE: This might not be right!!
            crb[si, j, :] = np.divide.outer(sensors[si].getCRB(),(abs(gard_true.g[j])**2)).T# # np.ones(np.size(row_ind))**2
        
        if plt:
            plt.subplot(np.floor(np.sqrt(Ns)),np.ceil(np.sqrt(Ns)),si+1)
            plt.plot(gard_true.r, gard_true.d, 'k.')
            plt.plot(gard_est.r, gard_est.d, 'rx')
            for i in range(Np):
                plt.text(gard_est.r[row_ind[i]], gard_est.d[row_ind[i]], (col_ind[i])+1, FontSize=7)
            plt.title('Sensor {}'.format(si+1) )
            plt.grid(True)
    return dr, dd, Nmiss, Nfa, crb, present

def get_true_ordered_list(garda_est, garda_true):
    Ns = len(garda_true)
    ordered_links=[[] for _ in range(Ns)]#List of Ordered lists linking object indices at sensors
    s=Ns-1
    R_mat = np.subtract.outer(garda_est[s].r,garda_true[s].r)**2 + np.subtract.outer(garda_est[s].d,garda_true[s].d)**2
    row_ind, col_ind = linear_sum_assignment(R_mat)
    link_ind = np.argsort(col_ind)
    ord_prev = np.argsort(row_ind[link_ind]) # Indices of garda_est in order of garda_true
    Np = len(garda_est[s].r)
    while s>0:
        gard_est = garda_est[s-1]
        gard_true = garda_true[s-1]
        R_mat = np.subtract.outer(gard_est.r,gard_true.r)**2 + np.subtract.outer(gard_est.d,gard_true.d)**2
        links = [] # [[] for l in range(len(r_c))] # Initialize List of Ns empty lists
        row_ind, col_ind = linear_sum_assignment(R_mat)
        link_ind = np.argsort(col_ind)
        rows_order = row_ind[link_ind]
        for j in range(Np):
            links.append(ob.link([rows_order[ord_prev[j]]]))

        ordered_links[s]=links # 3D list
        s=s-1
        Np = len(garda_est[s].r)
        ord_prev = np.argsort(rows_order)
    return ordered_links
