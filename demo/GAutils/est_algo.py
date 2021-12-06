"""
Created on Wed Apr 17 23:52:30 2019
Estimation algorithms for extracting Range, Doppler
@author: gupta
"""
from numpy import unravel_index
import numpy as np
from GAutils import objects as obt
#from numba import jit

def meth2(y, sensors, Nob, osf=[16, 16], pfa=1e-3, eps=0.005):
    garda = []
    residue_list = [[] for _ in sensors]
    for i, sensor in enumerate(sensors):
        gard = obt.gardEst()
        residue = np.linalg.norm(y[i])
        residue_list[i].append(residue)
        RD = estRD(y[i], sensor)
        tau = RD.get_cfar_thres(pfa)
        while True: #for ob in range(Nob):
            RD.detect_new(osf, gard)   # Subtract strong ob
#            print('R={}, D={}, G={}'.format(gard.r, gard.d, gard.g))
            residue = np.linalg.norm(RD.y)
            residue_list[i].append(residue)
#            print('Ref = {}'.format(time.time()-t))
            if Nob:
                if len(gard.r)>= Nob:
                    break
            else:
                if (abs(residue_list[i][-2]-residue)/residue_list[i][0]<eps): # Relative
                    if len(gard.r)>1:
                        gard.pop(len(gard.r)-1)
                    break
#                if (residue<tau): # ABSOLUTE 
#                    break
        RD.decouple_rd(gard)
        garda.append(gard)
    return garda
    
def nomp(y, sensors, Nob=[], osf=[16, 16], Nr=[1,3], pfa=1e-3, eps=0.005):
    """
    osf: Oversampling for Range,doppler
    Nr: Refinement steps for single, cyclic ref.
    TODO: CFAR based detection
    """
    garda = []
    residue_list = [[] for _ in sensors]
    for i, sensor in enumerate(sensors):
        gard = obt.gardEst()
#        resd = np.copy(y[i])
        residue = np.linalg.norm(y[i])
        residue_list[i].append(residue)
        RD = estRD(y[i], sensor)
        tau = RD.get_cfar_thres(pfa)
        while True: #for ob in range(Nob): # Replace with while for CFAR
#            t= time.time()
            RD.detect_new(osf, gard)   # Subtract strong ob
#            print('FFT = {}'.format(time.time()-t))
            ## DEbugging
            # (N0, N1) = resd.shape
            # mcs = sensor.mcs
            # rf = mcs.c / (2 * mcs.B)
            # df = mcs.c / (2 * mcs.tf * mcs.fc)
            # omg2 = 2*np.pi*gard.r/ rf / (N1) # Fast time omega
            # omg1 = 2*np.pi*gard.d/ df / (N0) # Slow time omega
            # tfa_ref2 = mcs.get_tfa(omg2, omg1 / N1, 1) / mcs.Ts
            # x_ref = np.exp(1j*tfa_ref2)/(tfa_ref2.size) # beat raw NOTE: check normalization
            # print(np.linalg.norm( y[i]-(resd+x_ref*gard.g*np.exp(1j*(N0*omg1+N1*omg2)/2))))
            ## ENd debug
#            t= time.time()
            idx = len(gard.r)-1 # last freq added to gard
            for sref in range(Nr[0]): #Single Refinenemt
                
                RD.refine_one(gard, Nr[0], idx)
                
            RD.refine_all(gard, Nr, 1)# cyclic refinement
            residue = np.linalg.norm(RD.y)
            residue_list[i].append(residue)
#            print('Ref = {}'.format(time.time()-t))
            if Nob:
                if len(gard.r)>= Nob:
                    break
            else:
                if (abs(residue_list[i][-2]-residue)/residue_list[i][0]<eps): # Relative
                    if len(gard.r)>1:
                        gard.pop(len(gard.r)-1)
                    break
#                if (residue<tau): # ABSOLUTE 
#                    break
        RD.decouple_rd(gard)
        garda.append(gard) # Add all frequencies identified
    return garda       
class estRD():
    def __init__(cls, y, sensor):
        mcs=sensor.mcs
        cls.mcs = mcs
        cls.sensor = sensor
        c = mcs.c
        cls.rf, cls.df = c / 2 / mcs.B, c / 2 / mcs.tf / mcs.fc
        cls.d2r = mcs.fc / mcs.ss
        cls.d2d = mcs.ss/mcs.fc*mcs.Ts*mcs.Ni/2
        cls.N0, cls.N1 =mcs.Nch, mcs.Ni
        cls.y = np.copy(y)
        cls.rN = np.sqrt(cls.N0* cls.N1)
        
        cls.tz = 1j *(np.arange(cls.N0)-cls.N0/2)
        cls.tx = 1j *(np.arange(cls.N1)-cls.N1/2)

    def detect_new(cls, osf, gard):
    #    c = 3e8
    #    rf, df = c / 2 / mcs.B, c / 2 / mcs.tf / mcs.fc
    #    d2r = mcs.fc / mcs.ss
        (N0, N1) = cls.N0, cls.N1
        fft1 = np.fft.fftshift(np.fft.fft2(cls.y, [N0*osf[0], N1*osf[1]]), axes=0)
        ind = unravel_index(np.abs(fft1).argmax(), fft1.shape)
        g1 = fft1[ind]/cls.rN
        r1 = cls.rf * (ind[1]) / osf[1] # cls.rf * (ind[1] - N1 * osf[1] / 2) / osf[1]
        d1 = cls.df * (ind[0] - N0 * osf[0] / 2) / osf[0]
        omg2 = 2*np.pi*r1/ cls.rf / (N1) # Fast time omega
        omg1 = 2*np.pi*d1/ cls.df / (N0) # Slow time omega    
        beat_recon = g1 * np.outer(np.exp(omg1*(cls.tz+1j*N1/2)),np.exp(omg2*(cls.tx+1j*N1/2)) ) / cls.rN
        # r1 = r1 - d2r * d1 # NOTE: Handle this
        cls.y = cls.y - beat_recon  # Subtract strong ob
        gard.add_Est(g1, 0, r1, d1)
        return
    
    def decouple_rd(cls, gard, rworld = False):
        """ Subtracts doppler coupling from range(fast time) freq."""
        if rworld:
            gard.d = gard.d/(1+cls.d2d)
            gard.r = gard.r - cls.d2r * gard.d # NOTE: Handle this
        return
    
    def get_cfar_thres(cls, pfa=1e-3):
        N2 = cls.rN**2
        tau = (
                cls.sensor.meas_std**2 ) * (np.log(N2) - np.log(np.log(1/(1-pfa))))
        return np.sqrt(tau)

    def refine_all(cls, gard, Nr, ord_flag=0):
        for cref in range(Nr[1]):
            if ord_flag:
                idxa = np.random.permutation(len(gard.r))
            else:
                idxa = range(len(gard.r))
            for idx in idxa:# Might randomize order in each cycle
                cls.refine_one(gard, Nr[0], idx)
        return
    
    # @profile
    # def refine_one(cls, gard, Nr, idx):#optimized
    #     for _ in range(Nr): # Repeat Nr times
    #         yr= cls.y
    #         r1, d1, g1 = gard.r[idx],gard.d[idx], gard.g[idx]
    #         N0, N1 = cls.N0, cls.N1
    #         rf, df = cls.rf, cls.df
    #         omg2 = 2*np.pi*r1/ rf / (N1) # Fast time omega
    #         omg1 = 2*np.pi*d1/ df / (N0) # Slow time omega    
    #         # Rescale g to get correct recon. (uncentered, )
    #         g1 = g1*np.exp(1j*(N0/2*omg1+N1/2*omg2)) # Uncomment
        
    #         tx, tz = cls.tx, cls.tz
    #         etx = np.exp(-omg2*tx)
    #         etz = np.exp(-omg1*tz)
    #     #    yrc = np.conj(yr)
    #         g2 = np.conj(g1)
    #         zy = etz @ yr
    #         yx = yr @ etx
    #         der2z = -2*np.real(g2 * np.inner(zy, (etx*(tx**2))))/cls.rN + 2*(np.abs(g1)**2)*(N0**2+2)/12
    #         der2x = -2*np.real(g2 * np.inner((etz*tz**2) , yx))/cls.rN + 2*(np.abs(g1)**2)*(N1**2+2)/12
    #         # der2xz = -2*np.real(g2 * np.linalg.multi_dot([etz*tz, yr, (etx*tx)]))/cls.rN
    #         if (der2x >0) or (der2z > 0):
    #             der2xz = -2*np.real(g2 * (etz*tz) @ yr @ (etx*tx))/cls.rN
    #             der1z = 2*np.real(g2*np.inner(zy, (etx*tx)))/cls.rN
    #             der1x = 2*np.real(g2*np.inner((etz*tz), yx))/cls.rN
    #         else:
    #             break
    #         # Update freq
    #         detder = der2x*der2z-der2xz**2
    #         if der2x > 0:
    #             omg1new = omg1 - (der1x*der2z-der1z*der2xz)/detder
    #         else:
    #                omg1new = omg1 # Do random pert. here
    #             # omg1 = omg1 - np.sign(der2x)*(1/4)*(2*np.pi/N0)*np.random.rand(1)
    #         if der2z > 0:
    #             omg2new = omg2 - (der1z*der2x-der1x*der2xz)/detder
    #         else:
    #            omg2new = omg2
    #             # omg2 = omg2 - np.sign(der2z)*(1/4)*(2*np.pi/N1)*np.random.rand(1)
        
    #         gard.r[idx] = omg2new * rf * (N1) /(2*np.pi)
    #         # Flip angular frequency, NOTE: remove if causes problems
    #         omg1new = np.sign(np.pi-omg1new)*np.minimum(omg1,2*np.pi-omg1new)
    #         gard.d[idx] = omg1new * df * (N0) / (2*np.pi)
        
    #         etx2 = np.exp(-omg2new*(tx+1j*N1/2))
    #         etz2 = np.exp(-omg1new*(tz+1j*N0/2))
    #         new_g = ( g1*(np.vdot(etx,etx2)*np.vdot(etz,etz2))/yr.size 
    #                 + np.linalg.multi_dot([etz2, yr, (etx2)])/cls.rN )
    #         cls.y = yr + (g1*np.outer(np.conj(etz),np.conj(etx)) 
    #                     - new_g*np.outer(np.conj(etz2),np.conj(etx2)))/cls.rN
    #         gard.g[idx] = new_g
    #     return
    
    # wrapper
    def refine_one(cls, gard, Nr, idx):#optimized
        yr= cls.y
        r1, d1, g1 = gard.r[idx],gard.d[idx], gard.g[idx]
        N0, N1 = cls.N0, cls.N1
        rf, df = cls.rf, cls.df
        omg2 = 2*np.pi*r1/ rf / (N1) # Fast time omega
        omg1 = 2*np.pi*d1/ df / (N0) # Slow time omega    
        # Rescale g to get correct recon. (uncentered, )
        # g1 = g1*np.exp(1j*(N0/2*omg1+N1/2*omg2)) # Uncomment
        rN = cls.rN
        tx, tz = cls.tx, cls.tz
            
        cls.y, omg1new, omg2new, new_g = cls.refine_one_all(yr, Nr, omg1, omg2,g1,N0,N1,tx,tz,rN)
        gard.r[idx] = omg2new * rf * (N1) /(2*np.pi)
        gard.d[idx] = omg1new * df * (N0) / (2*np.pi)
        gard.g[idx] = new_g
        return
    @staticmethod
    def refine_one_all(yr, Nr, omg1, omg2,g1,N0,N1,tx,tz,rN):#rev3
        for _ in range(Nr): # Repeat Nr times 
            # Rescale g to get correct recon. (uncentered, )
            g1 = g1*np.exp(1j*(N0/2*omg1+N1/2*omg2)) # Uncomment
        
            etx = np.exp(-omg2*tx)
            etz = np.exp(-omg1*tz)
        #    yrc = np.conj(yr)
            g2 = np.conj(g1)
            zy = etz @ yr
            yx = yr @ etx
            der1z = 2*np.real(g2*(zy @ (etx*tx)))/rN
            der1x = 2*np.real(g2*((etz*tz) @ yx))/rN
            der2z = -2*np.real(g2 * zy @ (etx*(tx**2)))/rN + 2*(np.abs(g1)**2)*(N0**2+2)/12
            der2x = -2*np.real(g2 * (etz*tz**2) @ yx)/rN + 2*(np.abs(g1)**2)*(N1**2+2)/12
            # der2xz = -2*np.real(g2 * np.linalg.multi_dot([etz*tz, yr, (etx*tx)]))/cls.rN
            der2xz = -2*np.real(g2 * (etz*tz) @ yr @ (etx*tx))/rN
            # Update freq
            detder = der2x*der2z-der2xz**2
        #    dw = np.array([(der1x*der2z-der1z*der2xz),(der1z*der2x-der1x*der2xz)])/detder
            if der2x > 0:
                omg1 = omg1 - (der1x*der2z-der1z*der2xz)/detder
            else:
        #            omg1 = omg1 # Do random pert. here
                omg1 = omg1 - np.sign(der2x)*(1/4)*(2*np.pi/N0)*np.random.random_sample()
            if der2z > 0:
                omg2 = omg2 - (der1z*der2x-der1x*der2xz)/detder
            else:
        #        omg2 = omg2
                omg2 = omg2 - np.sign(der2z)*(1/4)*(2*np.pi/N1)*np.random.random_sample()
        
            # Flip angular frequency, NOTE: remove if causes problems
            omg1 = np.sign(np.pi-omg1)*np.minimum(omg1,2*np.pi-omg1)
            # omg1 = (omg1 + np.pi) % (2 * np.pi) - np.pi
        
            etx2 = np.exp(-omg2*(tx+1j*N1/2))
            etz2 = np.exp(-omg1*(tz+1j*N0/2))
            new_g = ( g1*(np.vdot(etx,etx2)*np.vdot(etz,etz2))/yr.size 
                    + (etz2 @ yr @ etx2)/rN ) # np.linalg.multi_dot([etz2, yr, (etx2)])/rN )
            yr = yr + (g1*np.outer(np.conj(etz),np.conj(etx)) 
                        - new_g*np.outer(np.conj(etz2),np.conj(etx2)))/rN
            g1 = new_g
        return yr, omg1, omg2, new_g

    def refine_one2(cls, gard, Nr, idx):#rev2 slow
        yr= cls.y
        r1, d1, g1 = gard.r[idx],gard.d[idx], gard.g[idx]
        N0, N1 = cls.N0, cls.N1
        rf, df = cls.rf, cls.df
        omg2 = 2*np.pi*r1/ rf / (N1) # Fast time omega
        omg1 = 2*np.pi*d1/ df / (N0) # Slow time omega
        
        tx, tz = cls.tx, cls.tz
        etx = np.exp(omg2*tx)
        etz = np.exp(omg1*tz)
        x = np.outer(etz,etx)/cls.rN
        # Rescale g to get correct recon. (uncentered, )
        g1 = g1*np.exp(1j*(N0/2*omg1+N1/2*omg2)) # Uncomment
        ygx = np.conj(yr)*g1*x
        der1z = -2*np.real(np.sum(ygx@tz))
        der1x = -2*np.real(np.sum(tx@ygx))
        der2z = -2*np.real(np.sum(ygx@(tz**2))) + 2*(np.abs(g1)**2)*(N0**2+2)/12
        der2x = -2*np.real(np.sum((tx**2)@ygx)) + 2*(np.abs(g1)**2)*(N1**2+2)/12
        der2xz = -2*np.real(tx@ygx@tz) 
    #    # Refine xz
        # Update freq
        dw = np.linalg.lstsq([[der2x,der2xz],[der2xz,der2z]],[der1x,der1z])[0]
        if der2x > 0:
            omg1 = omg1 - dw[0]#der1x/der2x
        else:
            omg1 = omg1 # Do random pert. here
            omg1 = omg1 - np.sign(der2x)*(1/4)*(2*np.pi/N0)*np.random.rand(1)
        if der2z > 0:
            omg2 = omg2 - dw[1]#der1z/der2z
        else:
    #        omg2 = omg2
            omg2 = omg2 - np.sign(der2z)*(1/4)*(2*np.pi/N1)*np.random.rand(1)
        # Recompute beat (uncentered, normalized by full size)
        etx2 = np.exp(omg2*(tx+1j*N1/2))
        etz2 = np.exp(omg1*(tz+1j*N0/2))
        x_ref =  np.outer(etz2,etx2)/cls.rN # beat raw NOTE: check normalization
        
        gard.r[idx] = omg2 * rf * (N1) /(2*np.pi)
        # Flip angular frequency, NOTE: remove if causes problems
        omg1 = np.sign(np.pi-omg1)*np.amin([omg1,2*np.pi-omg1])
            
        gard.d[idx] = omg1 * df * (N0) / (2*np.pi)
        y_recon = yr + x * g1# beat reconstruction
        gard.g[idx] = np.sum(np.conj(x_ref)* y_recon)
        cls.y = y_recon - gard.g[idx]*x_ref 
        return
    def refine_one3(cls, gard, Nr, idx):#rev1 old
        yr,mcs= cls.y,cls.mcs
        r1, d1, g1 = gard.r[idx],gard.d[idx], gard.g[idx]
        (N0, N1) = yr.shape
        rf = mcs.c / (2 * mcs.B)
        df = mcs.c / (2 * mcs.tf * mcs.fc)
        omg2 = 2*np.pi*r1/ rf / (N1) # Fast time omega
        omg1 = 2*np.pi*d1/ df / (N0) # Slow time omega
        # tfa = (np.outer(np.arange(N0)-N0/2, np.ones(N1))*omg1
                # +np.outer(np.ones(N0), np.arange(N1)-N1/2)*omg2)
        tfa = mcs.get_tfa(omg2, omg1 / N1, 1) / mcs.Ts # center was 1
        tna_z = np.outer(np.ones(N0),np.arange(N1)-N1/2)
        tna_x = np.outer(np.arange(N1)-N1/2, np.ones(N0))
        # beat raw NOTE: norm by full size
        x = np.exp(1j*tfa)/np.sqrt(tfa.size) 
        dx_z = (1j * tna_z * x).flatten()
        d2x_z = -(np.square(tna_z)*x).flatten()
        dx_x = (1j * tna_x * x).flatten()
        d2x_x = -(np.square(tna_x)*x).flatten()
        # Rescale g to get correct recon. (uncentered, )
        g1 = g1*np.exp(1j*(N0/2*omg1+N1/2*omg2)) # Uncomment
        y_recon = yr + x * g1# beat reconstruction
        yrf = yr.flatten()
        # Refine xz
        der1x = -2*np.real(g1*np.vdot(yrf,dx_x))
        der2x = -2*np.real(g1*np.vdot(yrf,d2x_x)) + np.real(2*(np.abs(g1)**2)*np.vdot(dx_x,dx_x))
        der1z = -2*np.real(g1*np.vdot(yrf,dx_z))
        der2z = -2*np.real(g1*np.vdot(yrf,d2x_z)) + np.real(2*(np.abs(g1)**2)*np.vdot(dx_z,dx_z))
        # Update freq
        if der2x > 0:
            omg1 = omg1 - der1x/der2x
        else:
            omg1 = omg1 # Do random pert. here
        if der2z > 0:
            omg2 = omg2 - der1z/der2z
        else:
            omg2 = omg2
    #        omg2 = omg2 - np.sign(der2z)*(1/4)*(2*np.pi/N1)*np.random.rand(1)
       # Recompute beat (uncentered, normalized by full size)
        # tfa_ref = (np.outer(np.arange(N0), np.ones(N1))*omg1
                # +np.outer(np.ones(N0), np.arange(N1))*omg2)
        tfa_ref = mcs.get_tfa(omg2, omg1 / N1, 0) / mcs.Ts # center was 0
        x_ref = np.exp(1j*tfa_ref)/np.sqrt(tfa_ref.size) # beat raw NOTE: check normalization
        
        gard.r[idx] = omg2 * rf * (N1) /(2*np.pi)
        # Flip angular frequency, NOTE: remove if causes problems
        omg1 = np.sign(np.pi-omg1)*np.amin([omg1,2*np.pi-omg1])
            
        gard.d[idx] = omg1 * df * (N0) / (2*np.pi)
        gard.g[idx] = np.vdot((x_ref).flatten(), y_recon.flatten())
        cls.y = y_recon - gard.g[idx]*x_ref 
        return 
 
#%% Future work
def chirpz(x,A,W,M):
    """Compute the chirp z-transform.
    The discrete z-transform,
    X(z) = \sum_{n=0}^{N-1} x_n z^{-n}
    is calculated at M points,
    z_k = AW^-k, k = 0,1,...,M-1
    for A and W complex, which gives
    X(z_k) = \sum_{n=0}^{N-1} x_n z_k^{-n}    
    """
    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex,x.dtype) or np.issubdtype(np.float,x.dtype):
        dtype = x.dtype
    else:
        dtype = float

    x = np.asarray(x,dtype=np.complex)
    
    N = x.size
    L = int(2**np.ceil(np.log2(M+N-1)))

    n = np.arange(N,dtype=float)
    y = np.power(A,-n) * np.power(W,n**2 / 2.) * x 
    Y = np.fft.fft(y,L)

    v = np.zeros(L,dtype=np.complex)
    v[:M] = np.power(W,-n[:M]**2/2.)
    v[L-N+1:] = np.power(W,-n[N-1:0:-1]**2/2.)
    V = np.fft.fft(v)
    
    g = np.fft.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W,k**2 / 2.)

    return g
