#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:25:11 2018

@author: anantgupta
"""

from __future__ import division
# Add classes for Extended Targets
import numpy as np
import sympy as sp

from GAutils import proc_est as pr
from itertools import combinations
from GAutils import config as cfg
from GAutils import PCRLB as pcrlb

class FMCWprms:
    c = 3e8  # Speed of light

    def __init__(self, B=0.3e9, Ts=1 / 41e4, fc=6e10, Ni=32, Nch=32): # was 150M, 1.28M (1m, 0.7816m/s); (0.5G,0.82M)->(0.3m,0.5m/s)
        self.fc = fc
        self.B = B
        self.Ts = Ts
        self.Ni = Ni

        self.Nch = Nch
        self.ss = B / Ts / Ni
        self.tf = Nch * Ni * Ts
        Kr = Ts * self.ss * 4 * np.pi / self.c
        Kd = Ni* Ts * fc * 4 * np.pi / self.c
        self.FIMr =(Kr**2)*Nch*(Ni/6 * (2 * Ni**2 + 1))
        self.FIMd =(Kd**2)*Ni*(Nch/6*(2 * Nch**2 + 1))
#        self.x1, self.x2 = np.meshgrid(np.arange(self.Ni)-self.Ni/2,
#                             np.arange(self.Nch)-self.Nch/2)
    def get_tfa(self, ft=1, st=1, center=0):# center was 0
        self.x1, self.x2 = np.meshgrid(np.arange(self.Ni)-center*self.Ni/2,
                              np.arange(self.Nch)-center*self.Nch/2)
        tfa = self.Ts * (ft*self.x1 + self.Ni * st * self.x2)  # sampling time indices of frame
        return tfa


class PointTarget:
    # Add more parameters for target
    t = 1  # Time variable

    def __init__(self, xw, yw, vx, vy, proc_var=0.1, rcs=1):
        self.x = xw
        self.y = yw
        self.vx = vx
        self.vy = vy
        self.proc_var = proc_var
        self.state = [self.x,self.y,self.vx,self.vy]
        self.rcs = rcs


class Sensor:
    def __init__(self, xw, yw, vx=0, vy=0, ptx=1, mcs=FMCWprms(), meas_std = 0.0):
        self.x = xw
        self.y = yw
        self.vx = vx
        self.vy = vy
        self.ptx = ptx  # Tx power
        self.mcs = mcs  # FMCW parameters
        self.meas_std = meas_std
        self.fov = 1 # FOV sin(half beamwidth)
        self.crb = self.getCRB() 
#    print(FIM)
    def getCRB(self, scale=[1,1]):
        FIMr= self.mcs.FIMr
        FIMd = self.mcs.FIMd
        sigma= self.meas_std**2
        return (sigma/2)*np.array([scale[0]/FIMr, scale[1]/FIMd])

    def getnominalCRB(self, nom_snr=-20, scale=[1,1]):
        FIMr= self.mcs.FIMr
        FIMd = self.mcs.FIMd
        return (10**(-nom_snr/10)/2) * np.array([scale[0]/FIMr, scale[1]/FIMd])
   
class gardEst:
    def __init__(self):
        self.r = np.array([])# range
        self.d = np.array([])# doppler
        self.a = np.array([])# angle
        self.g = np.array([], dtype='complex')# complex gain
        self.ord = np.array([])

    def add_Est(cls, g, a, r, d):
        cls.ord = np.append(cls.ord, cls.r.shape)
        cls.r = np.append(cls.r, r)
        cls.d = np.append(cls.d, d)
        cls.a = np.append(cls.a, a)
        cls.g = np.append(cls.g, g)
        
    def pop(cls, i):
        cls.ord = np.delete(cls.ord,i)
        cls.r = np.delete(cls.r,i)
        cls.d = np.delete(cls.d,i)
        cls.a = np.delete(cls.a,i)
        cls.g = np.delete(cls.g,i)
        
class link: # Stores links to ranges in prev sensor and corr. vx's
    def __init__(self, indx=[], vxa=[], xa=[], llr=[]):
        self.indx = indx
        self.vxa = vxa
        self.xa = xa
        self.llr = llr
        
class State: #Linked list of states: mean, covariance
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.next = None       
            
class SignatureTracks: # collection of associated ranges[], doppler[] & estimated vx(scalar).
    # Precompute Constant matrices
    Zdict =dict()
    Widict =dict()
    udict = {}
    CRBdict = {}
    for Ninp in range(cfg.max_sensors): # Ninp in [2,Ns] NOTE: Should not be fixed
        Z_mat = np.eye(Ninp+2)
        Zt = Z_mat[0:-1,:]-Z_mat[1:,:] # consecutive
        Zt2 = np.zeros((int((Ninp+2)*(Ninp+1)/2),Ninp+2))
        for i, (j,k) in enumerate(combinations(range(Ninp+2),2)):
            Zt2[i,j]=1
            Zt2[i,k]=-1
        Wit = np.linalg.inv(0.5 * Zt @ Zt.T)
        Zdict[Ninp]=Zt
        Widict[Ninp] = Wit
        
    # For Extended Kalman Filter Initial covariance 
    Pinit_getter = pcrlb.CRBconverter()
    
    x, y, vx, vy, sx, rm, dm, sr, sd = sp.symbols('x y vx vy sx rm dm sr sd')

    r = sp.sqrt((x-sx)**2+y**2)
    d = ((x-sx)*vx+y*vy)/r
    # For EKF time update
    hk = [sp.lambdify([x,y,vx,vy,sx], r, "numpy"), sp.lambdify([x,y,vx,vy,sx], d, "numpy")] 
    # To Precompute H Matrix
    varl = [x, y, vx, vy]
    f =[[] for _ in range(2)] 
    for v1 in range(4):
        e = (r.diff(varl[v1]))
        # NOTE: Probe analytical expression for FIM element using e.expand()
        f[0].append(sp.lambdify([x,y,vx,vy,sx], e, "numpy") )
    for v1 in range(4):
        e = (d.diff(varl[v1]))
        # NOTE: Probe analytical expression for FIM element using e.expand()
        f[1].append(sp.lambdify([x,y,vx,vy,sx], e, "numpy") )
        
    def __init__(self, r, d, sindx, g=[]):
        self.r =[r]
        self.d = [d]
        self.g = [g]
        self.sindx = [sindx] # sensor index
        self.state_head = None # Linked List of states: Mean(3x1), D covariance matrix(3x3)
        self.state_end = None
        self.N=len(self.r)
        self.pid =[]# Obs order at sensor
        self.llr = 0 # Likelihood of observations
        self.gc = None # Geometric fitting error
        
    
    def get_Pinit(cls, sensors, target): # TODO: Get Pinit in principled manner
        xr,yr,vxr,vyr = target.state
        Am1 = np.zeros((4,4))
        for s, sensor in enumerate(sensors):       
            crb = sensor.getnominalCRB()
            cre = crb[0]
            cde = crb[1]
            F_mat = np.zeros((4,4))
            for v1 in range(4):
                for v2 in range(4):
                    F_mat[v1,v2] = cls.Pinit_getter.f[v1][v2](xr-sensor.x, yr, vxr, vyr, cre, cde)
            Am1[:,:] += F_mat
        Ami = np.linalg.inv(Am1)
        return Ami
    @profile    
    def get_newfit_error(cls, sensors, rnew, dnew, gnew, sidnew):
        # Reports geometry fitting error for given R,D pair
        rn = np.append(cls.r, rnew)
        dn = np.append(cls.d, dnew)
        # rn = np.hstack((cls.r, rnew))
        # dn = np.hstack((cls.d, dnew))
        # gn = np.hstack((cls.g, gnew))
        Me = rn*dn
        Me2 = rn*rn
        Ns = len(rn)
        sindx_new = np.append(cls.sindx,sidnew)
        keyval = tuple(sindx_new)
        L = np.array([sensors[si].x for si in sindx_new])
        if keyval in cls.CRBdict:
            CRB=cls.CRBdict[keyval]
        else:
            CRB = np.array([sensors[si].getnominalCRB() for i, si in enumerate(sindx_new)]) # Using nominal
            cls.CRBdict[keyval] = CRB
#        CRB = np.array([sensors[si].getCRB()/(abs(gn[i])**2) for i, si in enumerate(sindx_new)]) # Using est. gain
        # Get constants
        Z = cls.Zdict[Ns-2]
        Wi = cls.Widict[Ns-2]
        
        if keyval in cls.udict:
            u_vec=cls.udict[keyval]
        else:
            ZWZ = np.linalg.multi_dot([Z.T, Wi, Z])
            # Main estimator
            u_vec = ZWZ @ L/np.linalg.multi_dot([L.T, ZWZ, L])
            cls.udict[keyval] = u_vec
        # rd fitting
        v_hat = -Me @ u_vec # v_x estimate
        M1var = (np.sum( CRB * np.array([dn**2, rn**2]).T,1) 
        + np.prod(CRB,1) )
#        lb_vx_std = np.sqrt(M1var @ (u_vec**2)) # Std. dev in estimating vx
        # Fitting Error compuation (For each target)
        N_mat = Me @ Z.T + (v_hat *L) @(Z.T) # eta
        V_mat = N_mat @ Wi  # Optimal dual var.
        
        # r2 fitting
        x_hat = -(Me2 - ( L**2 )) @ u_vec/2
        M2var = (4*CRB[:,0] * np.array( rn**2) + CRB[:,0]**2)# Ignoring higher order terms
#        lb_x_std = np.sqrt(M2var @ (u_vec**2)/4) # std. dev in estimating x
        # Fitting Error compuation (For each target)
        N_mat2 = Me2 @ Z.T + 2*x_hat *L @ (Z.T) - ( L*L ) @ Z.T # eta
        U_mat = N_mat2 @ Wi # Optimal dual var.

        gc = ((cfg.rd_wt[0]*(V_mat@Z/2)**2/M1var + cfg.rd_wt[1]*((U_mat@Z/2)**2)/M2var))
        return sum(gc)
    
    def get_newfit_error_ekf(cls, sensors, rnew, dnew, gnew, sindx):
        
        Rk = np.diag(sensors[sindx].getnominalCRB())
        if cls.N>1: # Fetch previous State
            Stp = cls.state_end.mean
            if cls.N>2:
                Pp = cls.state_end.cov
            else:
                Pp = cls.get_Pinit(sensors, PointTarget(*Stp)) 
            Hk = np.zeros((2,4))
            for i in range(2):
                for j in range(4):
                    Hk[i,j] = cls.f[i][j](Stp[0],Stp[1],Stp[2],Stp[3],sensors[sindx].x)
            Ik = Hk @ Pp @ Hk.T + Rk # Innovation covariance (2x2)
            try:
                Kk = Pp @ Hk.T @ np.linalg.inv(Ik) # Kalman Gain (4x2)
                yk = np.array([rnew, dnew]) # Measurement
                yhk = np.array([cls.hk[i](Stp[0],Stp[1],Stp[2],Stp[3],sensors[sindx].x) for i in range(2)])
                Stn = Stp + Kk @ (yk - yhk)
                Pn = (np.eye(4) - Kk@Hk) @ Pp @ (np.eye(4) - Kk@Hk) + Kk @ Rk @ Kk.T
                
                return np.inner((yk - yhk), np.linalg.inv(Ik)@(yk - yhk))
            except: # If any degenerate case occurs
                return np.inf
        else: # Compute initial covariance
            return 1

    def get_partialest(cls, sensors, idx, gprev, modf):# returns estimate for given index of nodes
        rn = np.array([cls.r[si] for si in idx])
        dn = np.array([cls.d[si] for si in idx])
        Me = rn*dn
        Me2 = rn*rn
        Ns = len(idx)
        L = np.array([sensors[cls.sindx[si]].x for si in idx])
        CRB = np.array([sensors[cls.sindx[si]].getnominalCRB() for si in idx])# Using nominal
#        CRB = np.array([sensors[cls.sindx[si]].getCRB()/(abs(cls.g[si])**2) for si in idx])
        # Get constants
        Z = cls.Zdict[Ns-2]
        Wi = cls.Widict[Ns-2]
        # Main estimator
        u_vec = Z.T @ Wi @ Z @ L/(L.T @ Z.T @ Wi @ Z @ L)
        # rd fitting
        v_hat = -Me @ u_vec # v_x estimate
        M1var = (np.sum( CRB * np.array([dn**2, rn**2]).T,1) 
        + np.prod(CRB,1) )
#        lb_vx_std = np.sqrt(M1var @ (u_vec**2)) # Std. dev in estimating vx
        # Fitting Error compuation (For each target)
        N_mat = Me @ Z.T + (v_hat *L) @(Z.T) # eta
        V_mat = N_mat @ Wi  # Optimal dual var.
        g_nu = np.sqrt(np.sum(N_mat * V_mat) - np.sum((V_mat@Z)**2)/4)
        
        # r2 fitting
        x_hat = -(Me2 - ( L**2 )) @ u_vec/2
        M2var = (4*CRB[:,0] * np.array( rn**2) + CRB[:,0]**2)# Ignoring higher order terms
#        lb_x_std = np.sqrt(M2var @ (u_vec**2)/4) # std. dev in estimating x
        # Fitting Error compuation (For each target)
        N_mat2 = Me2 @ Z.T + 2*x_hat *L @ (Z.T) - ( L*L ) @ Z.T # eta
        U_mat = N_mat2 @ Wi # Optimal dual var.
        g_nu2 = np.sqrt(np.sum(N_mat2 * U_mat) - np.sum((U_mat@Z)**2)/4)
        gc = ((cfg.rd_wt[0]*(V_mat@Z/2)**2/M1var + cfg.rd_wt[1]*((U_mat@Z/2)**2)/M2var))
        
        xsa = x_hat - L
        y_est = np.sqrt(np.mean(Me2 - xsa **2))
        vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
        
        Stn = np.array([x_hat, y_est, v_hat, vy_est])
        Pn = np.diag([g_nu, g_nu2])
        new_state = State(Stn, Pn)
        if modf and abs(sum(gc))<gprev:# Update signature
            cls.gc=gc
            cls.r = rn
            cls.d = dn
            gn = [cls.g[si] for si in idx]
            cls.g = gn
            sindxn = [cls.sindx[si] for si in idx]
            cls.sindx = sindxn
            if cls.pid: 
                pidn = [cls.pid[si] for si in idx]
                cls.pid = pidn
            cls.N = Ns
            cls.state_end.next = new_state
            cls.state_end = new_state
            
        return new_state, sum(gc)
    
    def get_rd_fit_error(cls, sensors, mode='ls', all_flag=False):
        
        Me = np.array(cls.r) * np.array(cls.d)
        Me2 = np.array(cls.r) * np.array(cls.r)
        Ns = cls.N
        L = np.array([sensors[si].x for si in cls.sindx])
        CRB = np.array([sensors[si].getnominalCRB() for (si, gi) in zip(cls.sindx, cls.g)])
#        CRB = np.array([sensors[si].getCRB()/(abs(gi)**2) for (si, gi) in zip(cls.sindx, cls.g)])
        # Get constants
        Z = cls.Zdict[Ns-2]
        Wi = cls.Widict[Ns-2]
        if Ns<3: # Can't solve if Ns<3
            mode='ls'
        if mode=='ls':
            # Main estimator
            ZWZ = np.linalg.multi_dot([Z.T, Wi, Z])
            u_vec = ZWZ @ L/np.linalg.multi_dot([L.T ,ZWZ, L])
            # rd fitting
            v_hat = -Me @ u_vec # v_x estimate
            M1var = (np.sum( CRB * np.array([cls.d**2, cls.r**2]).T,1) 
            + np.prod(CRB,1) )
    #        lb_vx_std = np.sqrt(M1var @ (u_vec**2)) # Std. dev in estimating vx
            # Fitting Error compuation (For each target)
            N_mat = Me @ Z.T + (v_hat *L) @(Z.T) # eta
            V_mat = N_mat @ Wi  # Optimal dual var.
            g_nu = np.sqrt(np.sum(N_mat * V_mat) - np.sum((V_mat@Z)**2)/4)
            
            # r2 fitting
            x_hat = -(Me2 - ( L**2 )) @ u_vec/2
            M2var = (4*CRB[:,0] * np.array( cls.r**2) + CRB[:,0]**2)# Ignoring higher order terms
    #        lb_x_std = np.sqrt(M2var @ (u_vec**2)/4) # std. dev in estimating x
            # Fitting Error compuation (For each target)
            N_mat2 = Me2 @ Z.T + 2*x_hat *L @ (Z.T) - ( L*L ) @ Z.T # eta
            U_mat = N_mat2 @ Wi # Optimal dual var.
            g_nu2 = np.sqrt(np.sum(N_mat2 * U_mat) - np.sum((U_mat@Z)**2)/4)
            E1 = (V_mat@Z/2)**2#/M1var
            E2= (U_mat@Z/2)**2#/M2var
            # print (E1,E2)

        elif mode=='huber':
            if all_flag:
                Z = np.zeros((int(Ns*(Ns-1)/2),Ns))
                for i, (j,k) in enumerate(combinations(range(Ns),2)):
                    Z[i,j]=1
                    Z[i,k]=-1
            Me3 = cls.d*cls.d
            ve1 = np.sqrt((Z**2)@(Me2+Me3))
            ve2 = np.sqrt((Z**2)@(4*Me2))
            import cvxpy as cp
            beta_x = cp.Variable(1)
            beta_vx = cp.Variable(1)
            # Form and solve the Huber regression problem.
            cost = (cp.atoms.sum(cp.huber((2*beta_x*(L@Z.T/ve2) - (L*L)@Z.T/ve2 + Me2@Z.T/ve2), 0.1)))
#                   + cp.atoms.sum(cp.huber((beta_vx*(L@Z.T/ve1) + Me@Z.T/ve1), 0.01)))
            cp.Problem(cp.Minimize(cost)).solve()
            x_hat = beta_x.value
            
            cost = (cp.atoms.sum(cp.huber((beta_vx*(L@Z.T/ve1) + Me@Z.T/ve1), 0.1)))
            cp.Problem(cp.Minimize(cost)).solve()
            v_hat = beta_vx.value
                
#            E1 = 2*x_hat*(L) - (L*L) + Me2
#            E2 = -v_hat*(L) + Me 
        elif mode=='l1':
            if all_flag:
                Z = np.zeros((int(Ns*(Ns-1)/2),Ns))
                for i, (j,k) in enumerate(combinations(range(Ns),2)):
                    Z[i,j]=1
                    Z[i,k]=-1
            import cvxpy as cp
            beta_x = cp.Variable(1)
            beta_vx = cp.Variable(1)
        #     fit = norm(beta - beta_true)/norm(beta_true)
            cost = (cp.norm(2*beta_x*(L@Z.T) - (L*L)@Z.T + Me2@Z.T ,1)
                   +cp.norm(beta_vx*(L@Z.T) + Me@Z.T ,1))
#            cost = (cp.norm(2*beta_x*(L) - (L*L) + Me2 ,1)
#                   +cp.norm(beta_vx*(L) + Me ,1))
            constraints = [beta_x -L - cls.r <= 0,
                           -cls.r - beta_x + L <= 0]
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve()
            x_hat = beta_x.value
            v_hat = beta_vx.value
        elif mode=='l2':
            if all_flag:
                Z = np.zeros((int(Ns*(Ns-1)/2),Ns))
                for i, (j,k) in enumerate(combinations(range(Ns),2)):
                    Z[i,j]=1
                    Z[i,k]=-1
            import cvxpy as cp
            beta_x = cp.Variable(1)
            beta_vx = cp.Variable(1)
        #     fit = norm(beta - beta_true)/norm(beta_true)
            cost = (cp.norm(2*beta_x*(L@Z.T) - (L*L)@Z.T + Me2@Z.T ,2))
#                   +cp.norm(beta_vx*(L@Z.T) + Me@Z.T ,2))
            constraints = [beta_x -L - cls.r <= 0,
                           -cls.r - beta_x + L <= 0]
            prob = cp.Problem(cp.Minimize(cost))
            prob.solve()
            x_hat = beta_x.value
            cost2 = cp.norm(beta_vx*(L@Z.T) + Me@Z.T ,2)
            constraints = [beta_x -L - cls.r <= 0,
                           -cls.r - beta_x + L <= 0]
            prob = cp.Problem(cp.Minimize(cost2))
            prob.solve()
            v_hat = beta_vx.value
#            E1 = 2*x_hat*(L) - (L*L) + Me2
#            E2 = v_hat*(L) + Me
        xsa = x_hat - L
        y_est = np.sqrt(abs(np.mean(Me2 - xsa **2))) # TODO: handle negative value properly
        vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
        point = PointTarget(x_hat, y_est, v_hat, vy_est)
        E1 = (x_hat**2+y_est**2 - 2*x_hat*(L) + (L*L)) - Me2
        E2 = ((x_hat*v_hat + y_est*vy_est) -v_hat*(L)) - Me 
        return np.vstack((E1,E2)), point
    # @profile
    def get_state(cls, sensors):
        Me = cls.r * cls.d
        Me2 = cls.r * cls.r
        Ns = cls.N
        keyval = tuple(cls.sindx)
        L = np.array([sensors[si].x for si in cls.sindx])
        if keyval in cls.CRBdict:
            CRB=cls.CRBdict[keyval]
        else:
            CRB = np.array([sensors[si].getnominalCRB() for si in cls.sindx]) # Using nominal
            cls.CRBdict[keyval] = CRB
#       
        # CRB = np.array([sensors[si].getnominalCRB() for (si, gi) in zip(cls.sindx, cls.g)])
#        CRB = np.array([sensors[si].getCRB()/(abs(gi)**2) for (si, gi) in zip(cls.sindx, cls.g)])
        # Get constants
        Z = cls.Zdict[Ns-2]
        Wi = cls.Widict[Ns-2]
        # Main estimator
        if keyval in cls.udict:
            u_vec=cls.udict[keyval]
        else:
            ZWZ = np.linalg.multi_dot([Z.T, Wi, Z])
            # Main estimator
            u_vec = ZWZ @ L/np.linalg.multi_dot([L.T, ZWZ, L])
            cls.udict[keyval] = u_vec
        # rd fitting
        v_hat = -Me @ u_vec # v_x estimate
        M1var = (np.sum( CRB * np.array([cls.d**2, cls.r**2]).T,1) 
        + np.prod(CRB,1) )
#        lb_vx_std = np.sqrt(M1var @ (u_vec**2)) # Std. dev in estimating vx
        # Fitting Error compuation (For each target)
        N_mat = Me @ Z.T + (v_hat *L) @(Z.T) # eta
        V_mat = N_mat @ Wi  # Optimal dual var.
        g_nu = np.sqrt(np.sum(N_mat * V_mat) - np.sum((V_mat@Z)**2)/4)
        
        # r2 fitting
        x_hat = -(Me2 - ( L**2 )) @ u_vec/2
        M2var = (4*CRB[:,0] * np.array( cls.r**2) + CRB[:,0]**2)# Ignoring higher order terms
#        lb_x_std = np.sqrt(M2var @ (u_vec**2)/4) # std. dev in estimating x
        # Fitting Error compuation (For each target)
        N_mat2 = Me2 @ Z.T + 2*x_hat *L @ (Z.T) - ( L*L ) @ Z.T # eta
        U_mat = N_mat2 @ Wi # Optimal dual var.
        g_nu2 = np.sqrt(np.sum(N_mat2 * U_mat) - np.sum((U_mat@Z)**2)/4)
        cls.gc = ((cfg.rd_wt[0]*(V_mat@Z/2)**2/M1var + cfg.rd_wt[1]*((U_mat@Z/2)**2)/M2var))
        
        xsa = x_hat - L
        y_est = np.sqrt(abs(np.mean(Me2 - xsa **2))) # TODO: handle negative value properly
        vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
        
        Stn = np.array([x_hat, y_est, v_hat, vy_est])
        Pn = np.diag([g_nu, g_nu2])
        new_state = State(Stn, Pn)
        return new_state
    
    def add_sig(cls, newsig, st_indx, sensors):
        Np = cls.N
        # Adds obs from new_sig starting at st_indx 
        if newsig.N-st_indx>1:
            # compute x, y, vx from all obs can be used to update state)
            cls.r=np.append(cls.r, newsig.r[st_indx:])
            cls.d=np.append(cls.d, newsig.d[st_indx:])
            cls.g=np.append(cls.g, newsig.g[st_indx:])
            cls.sindx=np.append( cls.sindx, newsig.sindx[st_indx:])
            cls.N = cls.N+newsig.N-st_indx

            new_state = cls.get_state(sensors)
            
            if Np>1:
                cls.state_end.next = new_state
                cls.state_end = new_state
            else:
                cls.state_head = new_state
                cls.state_end = new_state
        else:
            cls.add_update3(newsig.r[st_indx], newsig.d[st_indx], newsig.g[st_indx], newsig.sindx[st_indx], sensors)
    
    def add_update3(cls, rs, ds, gs, sindx, sensors):
        # Dual cost method
        # TODO: maintain covariance matrix
        rp = cls.r
        dp = cls.d
        Np = cls.N
        sindxp = cls.sindx
        # compute x, y, vx from all obs can be used to update state)
        cls.r = np.append(cls.r, rs)
        cls.d = np.append(cls.d, ds)
        cls.g = np.append(cls.g, gs)
        cls.sindx = np.append(cls.sindx, sindx)
        cls.N = cls.N+1
            # Update previous covariance
        if Np > 1:
            new_state = cls.get_state(sensors)
            
            cls.state_end.next = new_state
            cls.state_end = new_state
        else:
            Pn = np.zeros((2,2))
            trg = pr.get_pos_from_rd(rp[0], rs, dp[0], ds, sindxp[0], sindx, sensors)
            if trg:
                Stn = np.array(trg.state)
                new_state = State(Stn, Pn)
                cls.state_head = new_state
                cls.state_end = new_state
            else:
                cls.N = cls.N-1
                cls.r = np.delete(cls.r, cls.N)
                cls.d = np.delete(cls.d, cls.N)
                cls.g = np.delete(cls.g, cls.N)
                cls.sindx = np.delete(cls.sindx, cls.N)
                raise ValueError('Attempted illegal add with {},{} at sensor ({},{})'.format(rp[0], rs, sindxp[0], sindx))
        
    def add_update_ekf(cls, rs, ds, gs, sindx, sensors):
        # Kalman Filter in Space (Oct 25, 2019)
        rp = cls.r
        dp = cls.d
        Np = cls.N
        # compute x, y, vx from all obs can be used to update state)
        sindxp = cls.sindx
        
        Fk = np.eye(4)
        Rk = np.diag(sensors[sindx].getnominalCRB())
        if Np>1: # Fetch previous State
            Stp = cls.state_end.mean
            Pp = cls.state_end.cov
            Hk = np.zeros((2,4))
            for i in range(2):
                for j in range(4):
                    Hk[i,j] = cls.f[i][j](Stp[0],Stp[1],Stp[2],Stp[3],sensors[sindx].x)
            Ik = Hk @ Pp @ Hk.T + Rk # Innovation covariance (2x2)
            Kk = Pp @ Hk.T @ np.linalg.inv(Ik) # Kalman Gain (4x2)
            yk = np.array([rs, ds]) # Measurement
            yhk = np.array([cls.hk[i](Stp[0],Stp[1],Stp[2],Stp[3],sensors[sindx].x) for i in range(2)])
            Stn = Stp + Kk @ (yk - yhk)
            Pn = (np.eye(4) - Kk@Hk) @ Pp @ (np.eye(4) - Kk@Hk) + Kk @ Rk @ Kk.T
        else: # Compute initial covariance
            trg = pr.get_pos_from_rd(rp[0], rs, dp[0], ds, sindxp[0], sindx, sensors)
            Pn = cls.get_Pinit(sensors, trg) 
            Stn = np.array(trg.state) # NOTE: Can give some prior here
        # Update previous covariance
        new_state = State(Stn, Pn)
        if Np > 1:
            cls.state_end.next = new_state
            cls.state_end = new_state
            curs = cls.state_head
            norm_const = np.diag(curs.cov)
            gc=[1]
            while curs is not None:
                gc.append(np.trace(curs.cov/norm_const)/2)
                curs = curs.next
            cls.gc = gc #(np.diag(Pn)/norm_const)*cls.N/2 # gc
        else:
            cls.state_head = new_state
            cls.state_end = new_state
        
        cls.r = np.append(cls.r, rs)
        cls.d = np.append(cls.d, ds)
        cls.g = np.append(cls.g, gs)
        cls.sindx = np.append(cls.sindx, sindx)
        cls.N = cls.N+1
        
        
    def add_update2(cls, rs, ds, sindx, lij, lc):
        # adding path using Kalman Filter
        rp = cls.r[-1]
        dp = cls.d[-1]
        cls.r = np.append(cls.r, rs)
        cls.d = np.append(cls.d, ds)
        cls.sindx = np.append(cls.sindx, sindx)
        # compute x, y, vx from kalman filter (NOTE: At this point all obs can be used to update state)
        xs = (rs**2 - rp**2) / (2 * lij) + lc # x estimated at current sensor
        ys = np.sqrt( (rs**2 + rp**2 - (lc**2)/2 - 2* (abs(xs-lc)**2))/2 ) # y estimated
        vxs = (rs*ds - rp*dp) / lij # vx same across all sensors
        vys = (rs*ds - vxs*xs) / ys # Note that this estimate is dependent on other estimates
        St = np.array([xs,vxs]) # Measurement=State 
#        cos_th = (xs / rs)**2
#        sin_th = (ys /rs)**2
#        del_mu = np.array([[cos_th, sin_th, 0],[0, 0, cos_th]]) # gradient of obs wrt state, 
#        if 0: # NOTE: add vx,vy dependence
#            del_mu[1,0] = (vxs*rs - xs*ds)/(rs**2) # unit is 1/sec
#            del_mu[1,1] = (vys*rs - ys*ds)/(rs**2)
        # del_geom = np.array([[2*rs,0,-2*rp,0],[2*rs,0,2*rp,0],[rs,ds,-rp,-dp]])
        del_geom = np.array([[2*rs,0,-2*rp,0],[rs,ds,-rp,-dp]])
        Rdvar = np.diag([0.01,0.01, 0.01, 0.01]) # Variance of range, doppler at sensor s, p (should depend on snr)
        R = del_geom @ Rdvar @ np.transpose(del_geom) # NOTE: covariance should be a function of actual/estimated position
        if cls.N>1: # Fetch previous State
            Pp = cls.state_end.cov
            Stp = cls.state_end.mean
            Kk = Pp @ np.linalg.inv(Pp + R)
        else: # Compute initial covariance
            Pp = np.zeros(4) 
            Stp = [0,0] # NOTE: Can give some prior here
            Kk = np.eye(2)
        # Update previous covariance
       
        Stf = Stp + Kk @ (St - Stp)
        Pf = (np.eye(2) - Kk) @ Pp @ (np.eye(2) - Kk) + Kk @ R @ Kk
        new_state = State(Stf, Pf)
        if cls.N > 1:
            cls.state_end.next = new_state
            cls.state_end = new_state
        else:
            cls.state_head = new_state
            cls.state_end = new_state
        
#        cls.vx= (N*cls.vx + vxs )/(N+1) # average the vx
        cls.N = cls.N+1

class obs_node:
    def __init__(self, g, a, r, d, oid, sid=0):
        self.g = g
        self.a = a
        self.r = r
        self.d = d
        self.oid = oid # Order in observation
        self.sid = sid
        self.lkf = []
        self.lkb = []
        self.visited = False
        self.used = None
        
    
    def insert_flink(cls, lk):
        cls.lkf.append(lk)
    def insert_blink(cls, lk):
        cls.lkb.append(lk)