#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:00:33 2019
Implement BP for solving association 
@author: anantgupta
"""
import numpy as np
from GAutils import proc_est as pr
#from GAutils import iter_prune as itpr 
from GAutils import objects as objt
from GAutils import gradient_methods as gm

def compute_linkage_error(sig, sensors):
    N = sig.N
    r = sig.r
    d = sig.d
    sindx = sig.sindx
    posM = [] #np.zeros(2, N*(N-1)/2)
    for i in range(N-1):
        for j in range(i+1,N):
            trg = pr.get_pos_from_rd(r[i], r[j], d[i], d[j], sindx[i], sindx[j], sensors)
            posM.append(trg.state)
    std = np.std(posM, axis = 0)
    mean = np.mean(posM, axis = 0)
    return mean, np.sum(std)#[0]+std[2]

def draw_line_reduction(garda_sel, sensors, scene, est_objects, alpha, plt, ttl=''):
    for i, gard in enumerate(garda_sel):
        yp = gard.r * gard.d
        xp = (2*i-3) * np.ones(len(yp))
        p1 = plt.plot(xp, yp, 'bo')
    x_val = np.array([sensor.x for sensor in sensors])#2*np.arange(4)-3
    if not alpha:
        alpha = np.ones(len(est_objects))
    else:
        alpha = 1-0.9*((alpha-min(alpha))/(max(alpha)-min(alpha)))
#    print(alpha)
    for (ob, ap) in zip(est_objects, alpha):
        r_est = np.sqrt(ob.x**2 + ob.y**2)
        d_est = (ob.x * ob.vx + ob.y * ob.vy)/r_est
        y_val = -x_val * ob.vx + r_est*d_est
        p2 = plt.plot(x_val, y_val, 'r-', alpha = ap)
    for obj in scene:
        r_true = np.sqrt(obj.x**2 + obj.y**2)
        d_true = (obj.x * obj.vx + obj.y * obj.vy)/r_true
        y_val = -x_val * obj.vx + r_true*d_true
        p3 = plt.plot(x_val, y_val, 'k.:')
    plt.grid(True)
    plt.xlabel('Sensors Abscissa (m)');plt.ylabel(r'Range x Doppler ($m^2/s$)');plt.title('Linear fit of Range, Doppler Product {}'.format(ttl))
    plt.legend([p3[0],p1[0],p2[0]],['True', 'Observation','Estimate'])
    return
def draw_line_reduction2(garda_sel, sensors, scene, est_objects, alpha, plt, ttl=''):
    for i, gard in enumerate(garda_sel):
        yp = gard.r * gard.r
        xp = (2*i-3) * np.ones(len(yp))
        p1 = plt.plot(xp, yp, 'bo')
    x_val = np.array([sensor.x for sensor in sensors])#2*np.arange(4)-3
    if not alpha:
        alpha = np.ones(len(est_objects))
    else:
        alpha = 1-0.9*((alpha-min(alpha))/(max(alpha)-min(alpha)))
#    print(alpha)
    for (ob, ap) in zip(est_objects, alpha):
        r_est = np.sqrt(ob.x**2 + ob.y**2)
        d_est = (ob.x * ob.vx + ob.y * ob.vy)/r_est
        y_val = -2*x_val*( ob.x-x_val/2) + r_est*r_est
        p2 = plt.plot(x_val, y_val, 'r-', alpha = ap)
    for obj in scene:
        r_true = np.sqrt(obj.x**2 + obj.y**2)
        d_true = (obj.x * obj.vx + obj.y * obj.vy)/r_true
        y_val = -2*x_val * (obj.x -x_val/2) + r_true*r_true
        p3 = plt.plot(x_val, y_val, 'k.:')
    plt.grid(True)
    plt.xlabel('Sensors Absicca (m)');plt.ylabel(r'Range * Range ($m^2$)');plt.title('Quadratic fitting of squared Range {}'.format(ttl))
    plt.legend([p3[0],p1[0],p2[0]],['True', 'Observation','Estimate'])
    return

def compute_linearity_error(sigs, sensors, w):
    Ns = len(sensors)
    erra =[]
    for i, sg in enumerate(sigs):
        x_val=np.array([sensors[si].x for si in sg.sindx])
        y_val = sg.r * sg.d
        center = np.mean(y_val)
        slope = np.sum((y_val-center) * x_val) / np.sum(x_val**2) #LS (MAE?)
        
        inst_slope = (y_val[1:]-y_val[0:-1] ) / (x_val[1:]-x_val[0:-1])
        angle_err = np.sum(abs((inst_slope - slope)/(1+inst_slope * slope))) # tan of slope difference
        angle_err += (Ns-sg.N) * w # w is penalty for missed detection
        
        err = np.sum(np.array((y_val - center) - slope * x_val )**2) # MSE error scales based on r,d (NOT a good error measure)
        err += (Ns-sg.N) * w # w is penalty for missed detection
        erra.append(angle_err)
    # NOTE: Cannot find position from these lines
    return erra

def compute_closeness(sigs, sensors, lamda, w):
    """Computes location straightness for estimates"""
    Ns = len(sensors)
    erra =[]
    phantoms = []
    # Precompute Constant matrices
    Zdict =dict()
    Widict =dict()
    for Ninp in range(Ns): # Ninp in [2,Ns]
        Z_mat = np.eye(Ninp+2)
        Zt = Z_mat[0:-1,:]-Z_mat[1:,:]
        Wit = np.linalg.inv(0.5 * Zt @ Zt.T)
        Zdict[Ninp]=Zt
        Widict[Ninp] = Wit
        
    for i, sg in enumerate(sigs):
        Me = sg.r * sg.d
        Me2 = sg.r * sg.r
        L = np.array([sensors[si].x for si in sg.sindx])
        Ns=sg.N

        # Get constants
        Z = Zdict[Ns-2]
        Wi = Widict[Ns-2]
        # Main estimator
        u_vec = Z.T @ Wi @ Z @ L/(L.T @ Z.T @ Wi @ Z @ L)
        # rd fitting
        v_hat = -Me @ u_vec # v_x estimate
#        M1var = cr**2 * dList**2 + cd**2 * rList**2 +cd**2 * cr**2
#        lb_vx_std = np.sqrt(M1var @ (u_vec**2)) # Std. dev in estimating vx
        # Fitting Error compuation (For each target)
        N_mat = Me @ Z.T + (v_hat *L) @(Z.T) # eta
        V_mat = N_mat @ Wi  # Optimal dual var.
        g_nu = np.sqrt(np.sum(N_mat * V_mat) - np.sum((V_mat@Z)**2)/4)
        
        # r2 fitting
        x_hat = -(Me2 + ( L**2 )/2) @ u_vec/2
#        M2var = 4 * cr**2 * rList**2 # Ignoring higher order terms
#        lb_x_std = np.sqrt(M2var @ (u_vec**2)/4) # std. dev in estimating x
        # Fitting Error compuation (For each target)
        N_mat2 = Me2 @ Z.T + 2*x_hat *L @ (Z.T) - ( L*L ) @ Z.T # eta
        U_mat = N_mat2 @ Wi # Optimal dual var.
        g_nu2 = np.sqrt(np.sum(N_mat2 * U_mat) - np.sum((U_mat@Z)**2)/4)
    
        angle_err = g_nu + lamda * g_nu2 # scalarized cost
        angle_err += (Ns-sg.N) * w # w is penalty for missed detection
        erra.append(angle_err)
        
        xsa = x_hat - L
        y_est = np.sqrt(np.mean(Me2 - xsa **2))
        vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
        phantoms.append(objt.PointTarget(x_hat, y_est, v_hat, vy_est))
    # NOTE: Cannot find position from these lines
    return erra, phantoms

#def iterative_tp_pruning(sigs, sensors, w, rd_wt):
#    centers=[]
#    sig_new = sigs
#    while 1:# for i in range(Nob):
##        lin_err = [compute_linkage_error(sg, sensors)[1] for sg in sig_new]
##        lin_err = compute_linearity_error(sig_new, sensors, w)
#        lin_err = [np.trace(sg.state_end.cov) for sg in sig_new]
#        rid = np.argmin(lin_err)
#        [new_pos, nlls_var] = gm.gauss_newton(sig_new[rid], sensors, sig_new[rid].state_end.mean , 5, rd_wt)
#        cluster_center = objt.PointTarget(new_pos[0], new_pos[1], new_pos[2], new_pos[3])
#        rindx = itpr.reduce_sigs(sig_new, cluster_center, sig_new[rid], rd_wt) # Remove signature of this obj
##        if cfg.debug_plots:
##            if max(llr_new)==min(llr_new):
##                llrnorm = llr_new*0 +1   
##            else:
##                llrnorm = (llr_new-min(llr_new))/(max(llr_new)-min(llr_new))
##            plt.subplot(3,3,len(centers))
##            [plt.quiver(obc.x, obc.y, obc.vx, obc.vy, color ='g', alpha = ln) for (obc,ln) in zip(ob_new,llrnorm)]
##            plt.quiver(cluster_center.x, cluster_center.y, cluster_center.vx, cluster_center.vy, color ='r')
##            plt.axis([-8.5,8.5,0,13])
##            plt.grid(True)#,plt.axis('equal')
##            plt.title(str(len(centers)))        
#        sig_new = [sig_new[ri] for ri in rindx]
#        centers.append(cluster_center)
#        if not (rindx):
#            break
#    return centers
