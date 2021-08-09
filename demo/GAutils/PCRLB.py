"""
Created on Wed Sep 26 14:35:46 2018
PCRLB calculation
@author: anantgupta
"""

import numpy as np
from GAutils import proc_est as pr
#import cvxpy as cp
import sympy as sp
import scipy as scp

def get_FIMrv(sensors, targets):
    Nt =  len(targets)
    Ns = len(sensors)
    CRLBr = np.zeros(( Nt, Ns))
    CRLBd = np.zeros(( Nt, Ns))
    rList = np.zeros(( Nt, Ns))
    dList = np.zeros(( Nt, Ns))
    for s, sensor in enumerate(sensors):
        mcs= sensor.mcs
        A = 4 * np.pi * mcs.ss / mcs.c
        B = 4 * np.pi * mcs.fc / mcs.c
        tfa0 = sensor.mcs.get_tfa()
        tna = (np.outer(np.ones([tfa0.shape[0],1]),tfa0[0])).flatten() - mcs.Ts * mcs.Ni/2
        tfa = tfa0.flatten() - mcs.tf/2
        part_der =[]
        for t, target in enumerate(targets):
            amp = pr.get_amplitude(sensor, target)/(sensor.meas_std/np.sqrt(tfa.size))#Per sample variance is scaled
            gard = pr.get_gard_true(sensor, target)
            rList[t,s]=gard.r
            dList[t,s]=gard.d
            beat = amp * np.exp(1j * (A * gard.r *tna + B * gard.d * tfa)) / np.sqrt(tfa.size) #- fbeat * tda / 2
            beat_r = A*tna * beat
            beat_d = B*tfa * beat
            if t>0:
                part_der = np.append(part_der, np.vstack([beat_r,beat_d]), axis = 0)
            else:
                part_der = np.vstack([beat_r,beat_d])
                
#                part_der = np.append(part_der, beat_d, axis = 1)
            
#                    Dr=B+A*tna-4*A*gard.r;
#                    Dv=B*t+A*t.*tn-4*A*r.*t;
#                    u11=Dr*(Dr-4*A).T
#                    u22=Dv*(Dv-4*A*(t.^2)).';
#                    u12=Dr*Dv.T-sum(4*A*t);
        ifish = 2*np.real( part_der @ np.conj(part_der.T) )
        try:
            CRLBs = np.diag(np.linalg.inv(ifish))
        except:
              print('FIM singular for sensor {}'.format(s))
              CRLBs = np.ones(2*Nt)*np.sqrt(np.finfo(float).max)
        CRLBr[:,s] = np.sqrt(CRLBs[0::2])
        CRLBd[:,s] = np.sqrt(CRLBs[1::2]) 
    return CRLBr, CRLBd, rList, dList

def get_FIMxy(sensors, targets):
    """Naive CRB evaluation, ignoring dependence of position, velocity"""
    Nt =  len(targets)
    Am1=np.zeros((4,4,Nt), dtype='complex128');
        
    for sensor in sensors:
        mcs= sensor.mcs
        A = 4 * np.pi * mcs.ss / mcs.c
        B = 4 * np.pi * mcs.fc / mcs.c
        tfa0 = sensor.mcs.get_tfa()
        tna = (np.outer(np.ones([tfa0.shape[0],1]),tfa0[0])).flatten() - mcs.Ts * mcs.Ni/2
        tfa = tfa0.flatten() - mcs.tf/2
        for t, target in enumerate(targets):
            amp = pr.get_amplitude(sensor, target)/(sensor.meas_std/ np.sqrt(tfa.size) )
            gard = pr.get_gard_true(sensor, target)
            r = gard.r
            xr = target.x - sensor.x
            yr = target.y - sensor.y
            beat = amp * np.exp(1j * (A * gard.r *tna + B * gard.d * tfa)) / np.sqrt(tfa.size) #- fbeat * tda / 2
            beat_diff = np.vstack([A*xr/r*tna * beat,A*yr/r*tna * beat,B*tfa*xr/r * beat,B*tfa*yr/r * beat] )
            Am1[:,:,t] += (beat_diff @ np.conj(beat_diff.T) )
    CRLBvx = []
    CRLBvy = []
    CRLBx = []
    CRLBy =[]
    CRB=[]

    for t in range(Nt):
        Ami1=np.linalg.inv(np.real(Am1[:,:,t]));    
        CRLBx.append(np.sqrt(Ami1[0,0]))
        CRLBy.append(np.sqrt(Ami1[1,1]))
        CRLBvx.append(np.sqrt(Ami1[2,2]))
        CRLBvy.append(np.sqrt(Ami1[3,3]))
        CRB.append(Ami1)

#    CRLBs = CRLBs * (CRLBs>0)
    # Unpack Ami2 to CRLB for each target
#    CRLBx = np.sqrt(CRLBs[0::2])
#    CRLBy = np.sqrt(CRLBs[1::2])
#    CRLBvx = np.sqrt(CRLBs[2::4])
#    CRLBvy = np.sqrt(CRLBs[3::4])
    return CRLBx, CRLBy, CRLBvx, CRLBvy, CRB

class CRBconverter:
    def __init__(self):
        x, y, vx, vy, rm, dm, sr, sd = sp.symbols('x y vx vy r d sr sd')

        r = sp.sqrt(x**2+y**2)
        d = (x*vx+y*vy)/r
        llr= 1/2*((rm-r)**2/sr + (dm-d)**2/sd) # LLR
        varl = [x, y, vx, vy]
        self.f =[[] for _ in range(4)] 
        for v1 in range(4):
            for v2 in range(4):
                e = (llr.diff(varl[v1],varl[v2])).subs([(rm,r),(dm,d)])
                # NOTE: Probe analytical expression for FIM element using e.expand()
                self.f[v1].append(sp.lambdify([x,y,vx,vy,sr,sd], e, "numpy") )
        # Can query each element of FIM matrix
    #    e11 = (llr.diff(x,x)).subs([(rm,r),(dm,d)])
    #    e12 = (llr.diff(x,y)).subs([(rm,r),(dm,d)])
    #    e21 = (llr.diff(y,x)).subs([(rm,r),(dm,d)])
    #    e22 = (llr.diff(y,y)).subs([(rm,r),(dm,d)])
    #    
    #    f11 = sp.lambdify([x,y,vx,vy,sr,sd], e11, "numpy") 
    #    f12 = sp.lambdify([x,y,vx,vy,sr,sd], e12, "numpy") 
    #    f21 = sp.lambdify([x,y,vx,vy,sr,sd], e21, "numpy") 
    #    f22 = sp.lambdify([x,y,vx,vy,sr,sd], e22, "numpy") 

    def get_CRBposvel_from_rd(self, cr, cd, sensors, targets):
        Nt =  len(targets)
    #    Am1=np.zeros((2,2,Nt), dtype='complex128');
        Am1=np.zeros((4,4,Nt));

        F_mat = np.zeros((4,4))
        for s, sensor in enumerate(sensors):       
            for t, target in enumerate(targets):
                gard = pr.get_gard_true(sensor, target)
                r = gard.r
                xr = np.squeeze(target.x - sensor.x)
                yr = np.squeeze(target.y - sensor.y)
                vxr = np.squeeze(target.vx - sensor.vx)
                vyr = np.squeeze(target.vy - sensor.vy)
    #            xr = np.asscalar(target.x - sensor.x)
    #            yr = np.asscalar(target.y - sensor.y)
    #            vxr = np.asscalar(target.vx - sensor.vx)
    #            vyr = np.asscalar(target.vy - sensor.vy)
                cre = cr[t,s]**2
                cde = cd[t,s]**2

                for v1 in range(4):
                    for v2 in range(4):
                        F_mat[v1,v2] = self.f[v1][v2](xr, yr, vxr, vyr, cre, cde)
                Am1[:,:,t] += F_mat          
        CRLBx = []
        CRLBy =[]
        CRLBvx = []
        CRLBvy =[]
        CRB=[]
        CRBp_rmse = []
        CRBp_area =[]
        CRBv_rmse = []
        CRBv_area =[]    
        for t in range(Nt):
            Ami1=np.linalg.inv(np.real(Am1[:,:,t]));    
            CRLBx.append(np.sqrt(Ami1[0,0]))
            CRLBy.append(np.sqrt(Ami1[1,1]))
            CRLBvx.append(np.sqrt(Ami1[2,2]))
            CRLBvy.append(np.sqrt(Ami1[3,3]))
            CRB.append(Ami1)
            CRBp_rmse.append(np.sqrt(np.trace(Ami1[0:2,0:2])))# Removed 1/2 factor
            CRBp_area.append(np.sqrt(np.linalg.det(Ami1[0:2,0:2])))
            CRBv_rmse.append(np.sqrt(np.trace(Ami1[2:4,2:4])))# Removed 1/2 factor
            CRBv_area.append(np.sqrt(np.linalg.det(Ami1[2:4,2:4])))
        return CRLBx, CRLBy, CRLBvx, CRLBvy, CRBp_rmse, CRBv_rmse

    def get_CRBposvelarea_from_rd(self, cr, cd, sensors, targets):
        Nt =  len(targets)
        Am1=np.zeros((4,4,Nt));

        F_mat = np.zeros((4,4))
        for s, sensor in enumerate(sensors):       
            for t, target in enumerate(targets):
                gard = pr.get_gard_true(sensor, target)
                r = gard.r
                xr = np.squeeze(target.x - sensor.x)
                yr = np.squeeze(target.y - sensor.y)
                vxr = np.squeeze(target.vx - sensor.vx)
                vyr = np.squeeze(target.vy - sensor.vy)
                cre = cr[t,s]**2
                cde = cd[t,s]**2
                for v1 in range(4):
                    for v2 in range(4):
                        F_mat[v1,v2] = self.f[v1][v2](xr, yr, vxr, vyr, cre, cde)
                Am1[:,:,t] += F_mat          
        CRBp_area =[]
        CRBv_area =[]    
        for t in range(Nt):
            Ami1=np.linalg.inv(np.real(Am1[:,:,t]));    

            CRBp_area.append(np.sqrt(np.linalg.det(Ami1[0:2,0:2])))
            CRBv_area.append(np.sqrt(np.linalg.det(Ami1[2:4,2:4])))
        return CRBp_area, CRBv_area

def get_CRBpos_from_r(cr, sensors, targets):
    Nt =  len(targets)
    Am1=np.zeros((2,2,Nt), dtype='complex128');
    for s, sensor in enumerate(sensors):       
        for t, target in enumerate(targets):
            gard = pr.get_gard_true(sensor, target)
            r = gard.r
            xr = target.x - sensor.x
            yr = target.y - sensor.y
            angle_mat = np.outer([xr/r,yr/r], [xr/r,yr/r])
            Am1[:,:,t] += angle_mat/(cr[t,s]**2)
    CRLBx = []
    CRLBy =[]
    for t in range(Nt):
        Ami1=np.linalg.inv(np.real(Am1[:,:,t]));    
        CRLBx.append(np.sqrt(Ami1[0,0]))
        CRLBy.append(np.sqrt(Ami1[1,1]))
    return CRLBx, CRLBy

#def get_CRBposvel_from_rd2(cr, cd, sensors, targets):
#    """Evaluation using hand calculated FIM failed"""
#    Nt =  len(targets)
#    Am1=np.zeros((4,4,Nt), dtype='complex128');
#    for s, sensor in enumerate(sensors):       
#        for t, target in enumerate(targets):
#            gard = pr.get_gard_true(sensor, target)
#            r = gard.r
#            r2 = r**2
#            xr = np.asscalar(target.x - sensor.x)
#            yr = np.asscalar(target.y - sensor.y)
#            vx = np.asscalar(target.vx - sensor.vx)
#            vy = np.asscalar(target.vy - sensor.vy)
#            angle_mat = np.outer([xr,yr], [xr,yr])
#            FIM0 = np.kron(np.diag([1/cr[t,s]**2, 1/cd[t,s]**2]), angle_mat)
#            M_mat = np.outer([xr,yr], [vx,vy])
#            Z2=np.zeros([2,2])
#
#            inter_mat = np.array([[yr*vy, yr*vx-xr*vy],[xr*vy-yr*vx, xr*vx]])
#            FIM1 = np.block([[-M_mat*inter_mat, M_mat @ np.diag([yr**2, xr**2])],[M_mat, Z2]])/cd[t,s]**2
#            S_mat =np.array([[1,-1],[-1,1]])
#            FIM2 = np.block([[S_mat@np.outer([(xr*vx)**2,(yr*vy)**2],[1,1])-np.diag([(xr*vx)**2,(yr*vy)**2]), np.diag([xr**2,yr**2])@ np.outer([yr,xr],[vx,vy])],[Z2,Z2]] )/cd[t,s]**2
#            FIM = 1/r2 * (FIM0 + FIM1/r2 + FIM2/r2)
#            Am1[:,:,t] += FIM
#    CRLBvx = []
#    CRLBvy = []
#    CRLBx = []
#    CRLBy =[]
#    for t in range(Nt):
#        Ami1=np.linalg.inv(np.real(Am1[:,:,t]));    
#        CRLBx.append(np.sqrt(Ami1[0,0]))
#        CRLBy.append(np.sqrt(Ami1[1,1]))
#        CRLBvx.append(np.sqrt(Ami1[2,2]))
#        CRLBvy.append(np.sqrt(Ami1[3,3]))
#    return CRLBx, CRLBy, CRLBvx, CRLBvy

    
def get_fim_track(sensor, target, m, AbsPos):
    # Calculates FIMs along a track with 
    # m: iteration number
    # AbsPos is position m intervals ago, (initial, At m=0)
    amp = 1  # Should be related to distance between sensor & target
    rho = pr.get_amplitude(sensor,target)**2/sensor.meas_var
    F = np.eye(4)
    F[0,2] = sensor.mcs.tf
    F[1,3] = sensor.mcs.tf
    Q = np.eye(4)*target.proc_var
    D12 = np.matmul(np.transpose(F),np.inverse(Q))
    D21 = np.transpose(D12)
    D11 = np.matlmul(D12,F)

def get_PCRLB(sensor, target, m, AbsPos, Np):
    # m: iteration number
    # AbsPos is position m intervals ago
    amp = 1  # Should be related to distance between sensor & target
    rho = pr.get_amplitude(sensor,target)**2/sensor.meas_var
    F = np.eye(4)
    F[0,2] = sensor.mcs.tf
    F[1,3] = sensor.mcs.tf
    Q = np.eye(4)*target.proc_var
    D12 = np.matmul(np.transpose(F),np.inverse(Q))
    D21 = np.transpose(D12)
    D11 = np.matlmul(D12,F)
    Gp, AbsPnext = 0,0 # WARNING: Add function HERE!!
    D22 = np.inverse(Q) + get_FIMxy()
    if m>0:
        
        fim = D22 - np.matmul(np.matmul(D21,np.inverse(+D11)),D12)
    else:
        fim = np.zeros([4,4])# No prior information at sensor!
    return fim

#def get_gap(Merr, C):
#    """Return the MSE in fitting Nobs lines to measurements.
#        Keyword arguments:
#            Merr -- Observed measurements [Nobs,Ns]
#            imag -- Predefined slopes of lines. Size [Nobs,Ns]
#        Returns:
#            err_ind -- Fitting Error (std) for all lines.
#    """
#    # c should be aligned with Merr columns
#    (Nobs, Ns) = Merr.shape
#    Z_mat = np.eye(Ns)
#    Z_sum = Z_mat[:,0:-1]+Z_mat[:,1:]
#    Merrd = Merr @ Z_sum # Variance of difference
#    # Create two scalar optimization variables.
#    v1 = cp.Variable(Nobs) # vector var
#    v2 = cp.Variable(Nobs) # vector var
#    v3 = cp.Variable(Nobs) # vector var
#    u1=cp.Parameter()
#    u2=cp.Parameter()
#    # Create two constraints.
##    constraints = [x[0] + 2*x[1] <= u1,
##                   x[0] - 4*x[1] <= u2,
##                   x[0] + x[1] >= -5]
#    # Form objective.
#    #obj = cp.Maximize(cp.quad_form(x, Q) -x[0])
#    obj = cp.Maximize( v1.T @ Merr[:,0] + v2.T @ Merr[:,1] + v3.T @ Merr[:,2] 
#                    - 1/4 * cp.atoms.sum_squares(cp.atoms.affine.vstack.vstack([v1, v2-v1, v3-v2, -v3])) 
#                    - (v1.T@C[:,0]+v2.T@C[:,1]+v3.T@C[:,2]) )
#    
#    # Form and solve problem.
#    u1.value =-2
#    u2.value =-3
#    prob = cp.Problem(obj)
#    
#    
#    prob.solve()
#    # Q1 Optimal value
#    v1v = np.array(v1.value)
#    v2v = np.array(v2.value)
#    v3v = np.array(v3.value)
#    # Errors for fitting individual lines
#    err_idv = np.sqrt(
#            np.sum(np.multiply(np.hstack([v1v,v2v,v3v]) , (Merrd - C)), axis =1)
#            -1/2*(v1v**2 +v2v**2+v3v**2-v1v*v2v-v2v*v3v).flatten() )
#    # Q2 variables
##    print(v1.value)
#    # Print dual variable 
##    for i in range(3):
##        print(constraints[i].dual_value)
##    l1 = constraints[0].dual_value
##    l2 = constraints[1].dual_value
#    #for i in range(3):
#    #    for j in range(3):
#    #        d1 = (i-1)*0.1
#    #        d2 = (j-1)*0.1
#    #        u1.value=-2+d1
#    #        u2.value=-3+d2
#    #        prob = cp.Problem(obj, constraints)
#    #        prob.solve()
#    #        p_ex = obj.value
#    #        p_pr = p0 - d1*l1 - d2*l2
#    #        print('pex',p_ex,', ppr',p_pr,', Diff =', p_ex-p_pr )
#    return err_idv

def get_QBound(cr, cd, rList, dList, sensors):
    """ Compute lower bound, estimates by solving dual maximization"""
    Me = rList*dList
    Me2 = rList * rList
    L = np.array([sensor.x for sensor in sensors])    
    Ns=len(sensors)
    # Constant matrices
    Z_mat = np.eye(Ns)
    Z = Z_mat[0:-1,:]-Z_mat[1:,:]
#    U = 0.5 * (Z_mat[0:-1,:]+Z_mat[1:,:])
    Wi = np.linalg.inv(0.5 * Z @ Z.T)

    # Main estimator
    u_vec = Z.T @ Wi @ Z @ L/(L.T @ Z.T @ Wi @ Z @ L)
    # rd fitting
    v_hat = -Me @ u_vec # v_x estimate
    M1var = cr**2 * dList**2 + cd**2 * rList**2 +cd**2 * cr**2
    lb_vx_std = np.sqrt(M1var @ (u_vec**2)) # Std. dev in estimating vx
    # Fitting Error compuation (For each target)
    N_mat = Me @ Z.T + np.outer(v_hat , (Z @ L)) # eta
    V_mat = N_mat @ Wi  # Optimal dual var.
    g_nu = np.sqrt(np.sum(N_mat*V_mat, axis =1) - np.sum((V_mat@Z)**2 , axis =1)/4)
    
    # r2 fitting
    o1v = np.ones_like(Me2[:,0])
    x_hat = -(Me2 + np.outer(o1v , L**2 )/2) @ u_vec/2
    M2var = 4 * cr**2 * rList**2 # Ignoring higher order terms
    lb_x_std = np.sqrt(M2var @ (u_vec**2)/4) # std. dev in estimating x
    # Fitting Error compuation (For each target)
    N_mat2 = Me2 @ Z.T + 2*np.outer(x_hat , (Z @ L)) - 2*np.outer(o1v , L**2 )/2 @ Z.T # eta
    U_mat = N_mat2 @ Wi # Optimal dual var.
    g_nu2 = np.sqrt(np.sum(N_mat2*U_mat, axis =1) - np.sum((U_mat@Z)**2 , axis =1)/4)
    return lb_vx_std, lb_x_std, g_nu, g_nu2


def ZZBrv(sensor, target):
    mcs = sensor.mcs
    N1 = mcs.Ni
    N2 = mcs.Nch
    Rmax = mcs.c/2/mcs.B*N1
    vmax = mcs.c/2/mcs.fc/mcs.Ts/mcs.Ni;
#     sigma = np.sqrt(10**(-snr/10))
    sigma = (sensor.meas_std)/pr.get_amplitude(sensor, target)
    Distance = lambda x: np.sqrt(N1*N2*(1-abs( np.sin(N1*x/2)/(N1*np.sin(x/2)) )))
    Distance2 = lambda x: np.sqrt(N1*N2*(1-abs( np.sin(N2*x/2)/(N2*np.sin(x/2)) )))
    #NUMERICAL INTEGRATION
    z = scp.integrate.quad(lambda x: x*scp.stats.norm.sf(Distance(x)/sigma), 0, np.pi, epsabs=1e-10)
    z2 = scp.integrate.quad(lambda x: x*scp.stats.norm.sf(Distance2(x)/sigma), 0, np.pi, epsabs=1e-10)
    ZZBr = np.sqrt(z[0])*(Rmax/2/np.pi)
    ZZBd = np.sqrt(z2[0])*(vmax/2/np.pi)
    [CRBr, CRBd,_,_] = get_FIMrv([sensor], [target])
#     plt.plot(y,y*scp.stats.norm.sf(Distance(y)/sigma),'r')
    return max(CRBr,ZZBr), max(CRBd,ZZBd) # ZZB not evaluated correctly at high SNR