import numpy as np
import sympy as sp
from scipy.optimize import least_squares

class Jacobians():
    """ Symbolic computation of derivates"""
    x, y, vx, vy, r, d, rm, dm, sr, sd = sp.symbols('x y vx vy r d rm dm sr sd')
    r = sp.sqrt(x**2+y**2)
    d = (x*vx+y*vy)/r
    llr= 1/2*((rm-r)**2/sr + (dm-d)**2/sd) # LLR
    varl = [x, y, vx, vy]
    cost_r = -(rm-r)/sr 
    cost_d = -(dm-d)/sd
    f1r=[]
    f1d=[]
    for v in range(4):
        e1r= (cost_r.diff(varl[v]))
        e1d= (cost_d.diff(varl[v]))
        f1r.append(sp.lambdify([x,y,vx,vy,rm, dm,sr,sd], e1r, "numpy"))
        f1d.append(sp.lambdify([x,y,vx,vy,rm, dm,sr,sd], e1d, "numpy"))
          
    def __init__(self, sg, sensors):
#        self.f2 =[[] for _ in range(4)]
#        for v1 in range(4):
#            for v2 in range(4):
#                self.e = (self.llr.diff(self.varl[v1],self.varl[v2])).subs([(self.rm,robs),(self.dm,dobs)])
#                # NOTE: Probe analytical expression for FIM element using e.expand()
#                self.f2[v1].append(sp.lambdify([self.x,self.y,self.vx,self.vy,self.sr,self.sd], self.e, "numpy") )
        self.rm = sg.r
        self.dm = sg.d
        self.M=sg.N
        self.sensx = [sensors[sid].x for sid in sg.sindx]
        if 0:
            crbvec = [sensors[sid].getCRB() for sid in sg.sindx]
            self.sr = np.sqrt([ cr[0]/abs(g)**2 for (cr, g) in zip(crbvec,sg.g)])
            self.sd = np.sqrt([ cr[1]/abs(g)**2 for (cr, g) in zip(crbvec,sg.g)])
        else:
            crbvec = [sensors[sid].getnominalCRB() for sid in sg.sindx]
            self.sr = np.sqrt([ cr[0] for cr in crbvec])
            self.sd = np.sqrt([ cr[1] for cr in crbvec])
            
        self.J_mat = np.zeros((2*sg.N, 4))
    
    def get_J(cls, pos):
        (x, y) = (pos[0], pos[1])
        (vx, vy) = (pos[2], pos[3])
        xvec=x-cls.sensx
        try:
            for v1 in range(4):
                cls.J_mat[:cls.M,v1] = [cls.f1r[v1](xr, y, vx, vy, rm, dm, sri, sdi)
                        for (xr, rm, dm, sri, sdi) in zip(xvec,cls.rm,cls.dm,cls.sr,cls.sd)]
                cls.J_mat[cls.M:,v1] = [cls.f1d[v1](xr, y, vx, vy, rm, dm, sri, sdi)
                        for (xr, rm, dm, sri, sdi) in zip(xvec,cls.rm,cls.dm,cls.sr,cls.sd)]
        except:
            print(xvec),print(cls.rm),print(cls.dm),print(cls.sr),print(cls.sd)
        return cls.J_mat
#            # NOTE: Probe analytical expression for FIM element using e.expand()
#            self.f1r.append(sp.lambdify(
#                    [self.x,self.y,self.vx,self.vy,self.rm, self.dm,self.sr,self.sd], e1r[v], "numpy"))
#            self.f1d.append(sp.lambdify(
#                    [self.x,self.y,self.vx,self.vy,self.rm, self.dm,self.sr,self.sd], e1d[v], "numpy") )
            
def gauss_newton(sg, sensors, init, itera, w=[1,0]):
    (x, y) = (init[0], init[1])
    (vx, vy) = (init[2], init[3])
    r_obs = sg.r
    d_obs = sg.d
    M = sg.N # Number of residuals
    xvec = x - [sensors[sid].x for sid in sg.sindx]
    
    yvec = y * np.ones(len(xvec))
    rvec = r_eval([xvec, yvec])
    dvec = d_eval([xvec, yvec, vx, vy])
    Jc = Jacobians(sg, sensors)
    F_mat = np.zeros((4,4))
    
    if itera > 0: # Do Gauss-newton refinement
        if 0:
            J = np.column_stack((xvec/rvec, yvec/rvec))
            out = [x,y] - np.linalg.pinv(J) @ (rvec - r_obs)
        #    out2 = init[2:3] - np.linalg.inv(J.T @ J) @ J.T @ (dvec - d_obs)
            out2 = np.linalg.pinv(J) @ d_obs
            pos = np.concatenate([out,out2])
        else:
            J_mat = Jc.get_J(init)
#            for v1 in range(4):
#                J_mat[:4,v1] = [Jc.f1r[v1](xr, y, vx, vy, cr[0]/abs(g)**2, cr[1]/abs(g)**2)
#                    for (xr, cr, g) in zip(xvec,crbvec,sg.g)]
#                J_mat[4:,v1] = [Jc.f1d[v1](xr, y, vx, vy, cr[0]/abs(g)**2, cr[1]/abs(g)**2)
#                    for (xr, cr, g) in zip(xvec,crbvec,sg.g)]
#                for v2 in range(4):# compute using hessian
#                    F_mat[v1,v2] = Jc.f2[v1][v2](xr, yr, vxr, vyr, cre, cde)
            try:
                pos = init - np.linalg.pinv(J_mat) @ np.hstack(((rvec - r_obs)/Jc.sr, (dvec -d_obs)/Jc.sd))
            except:
                print('Position refinement error')
                pos = init
        var = 0
    
#        print('x:{}, y:{}'.format(out[0],out[1]))
        return gauss_newton(sg, sensors, pos, itera-1, w)
    else: #Compute variance wrt observations
#        print('Done.')
        var = np.linalg.norm([w[0]*(r_obs-rvec), w[1]*(d_obs-dvec)])# Weight range mismatch 10x NOTE: Can weight based on x,y,vx,vy separately
        return init, var

def lm_refine(sg, sensors, init, itera, w=[1,0]):
    """LM algorithm (GD+NLLS) for position refinement"""
    r_obs = sg.r
    d_obs = sg.d
    if 0:
        crbvec = [sensors[sid].getCRB() for sid in sg.sindx]
        sr = np.sqrt([ cr[0]/abs(g)**2 for (cr, g) in zip(crbvec,sg.g)])
        sd = np.sqrt([ cr[1]/abs(g)**2 for (cr, g) in zip(crbvec,sg.g)])
    else:
        crbvec = [sensors[sid].getnominalCRB() for sid in sg.sindx]
        sr = np.sqrt([ cr[0] for cr in crbvec])
        sd = np.sqrt([ cr[1] for cr in crbvec])
    sensx = [sensors[sid].x for sid in sg.sindx]
    
    def get_res(init):
        (x, y) = (init[0], init[1])
        (vx, vy) = (init[2], init[3])
        xvec = x - sensx
        r = np.sqrt(xvec**2+y**2)
        d = (xvec*vx+y*vy)/r
        cost_r = (r_obs-r)/sr 
        cost_d = (d_obs-d)/sd
        return np.hstack((cost_r, cost_d))
    result1 = least_squares(get_res, init, method='lm', xtol=1e-3)
    return result1.x, result1.cost

def r_eval(pos, sxy = []):
    """
        pos =[x,y,vx,vy] absolute target position
        sxy = [sensor.x, sensor.y]
    """
    if not sxy:        
        sxy = [0,0]
    return np.sqrt(np.square(pos[0] - sxy[0])+np.square(pos[1] - sxy[1]))

def d_eval(pos, sxy=[]):
    """
    Needs all 4 states in pos
    """
    pos2 = pos 
    if sxy:        
        pos2[0] = pos2[0] - sxy[0] # relative x
        pos2[1] = pos2[1] - sxy[1] # relative y
    return (pos2[0]*pos2[2]+pos2[1]*pos2[3])/r_eval(pos2)

# Huber 
def huber(sg, sensors, init, itera, w=[1,0]):
    import cvxpy as cp
    beta_x = cp.Variable(1)
    beta_vx = cp.Variable(1)
    Me = sg.r * sg.d
    Me2 = sg.r * sg.r
    L = np.array([sensors[sid].x for sid in sg.sindx])
    Z_mat = np.eye(sg.N)
    Z = Z_mat[0:-1,:]-Z_mat[1:,:]
    # Form and solve the Huber regression problem.
    cost = (cp.atoms.sum(cp.huber(2*beta_x*(L@Z.T) - (L*L)@Z.T + Me2@Z.T, 5))
           + cp.atoms.sum(cp.huber(beta_vx*(L@Z.T) + Me@Z.T, 5)))
    cp.Problem(cp.Minimize(cost)).solve()
    x_hat = beta_x.value
    v_hat = beta_vx.value
    # Compute yparams
    xsa = x_hat - L
    y_est = np.sqrt(np.mean(Me2 - xsa **2))
    vy_est = np.mean(Me - v_hat*xsa) / y_est # Estimated using other estimates
    
    return [x_hat, y_est, v_hat, vy_est], cost.value