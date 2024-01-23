import numpy as np
from main3 import lissajous
from tqdm import tqdm
from casadi import *

p_info = [[-3,-3], [3,3], 0.5] # lower, upper bounds and resolution
theta_info = [-np.pi, np.pi, 0.5]# lower, upper bounds and resolution
v_info = [0, 1, 0.1] 
omega_info = [-1, 1, 0.1]
pmin, pmax, pres = p_info
thetamin, thetamax, thetares = theta_info 
vmin, vmax, vres = v_info 
omegamin, omegamax, omegares = omega_info

# Discretization ####
P = {}
P['res'] = pres
P['size'] = (np.ceil((np.array(pmax)-np.array(pmin))/pres) + 1).astype(np.int64)
P['x'] = pmin[0]+pres*np.array(list(range(P['size'][0])))
P['y'] = pmin[1]+pres*np.array(list(range(P['size'][1])))
THETA = {}
THETA['res'] = thetares
THETA['size'] = (np.ceil((thetamax-thetamin)/thetares) + 1).astype(np.int64)
THETA['theta'] = thetamin+thetares*np.array(list(range(THETA['size'])))
Vel = {}
Vel['res'] = vres
Vel['size'] = (np.ceil((vmax-vmin)/vres) + 1).astype(np.int64)
Vel['v'] = vmin+vres*np.array(list(range(Vel['size'])))
OMEGA = {}
OMEGA['res'] = omegares
OMEGA['size'] = (np.ceil((omegamax-omegamin)/omegares) + 1).astype(np.int64)
OMEGA['omega'] = omegamin+omegares*np.array(list(range(OMEGA['size'])))
#####

nx = P['size'][0]
ny = P['size'][1]
ntheta = THETA['size']
x_values = P['x']
y_values = P['y']
theta_values = THETA['theta']
n_v = Vel['size']
n_omega = OMEGA['size'] 

N_t = 100
time_step = 0.5
gamma = 0.9
Q = np.array([[10, 0], [0, 10]])
R = np.array([[1, 0], [0, 1]])
q = 1

########################################################################
def g_func(time_step, cur_t, cur_err_state, control, noise = False):
    cur_ref_state = np.array(lissajous(cur_t))
    next_ref_state = np.array(lissajous(cur_t+1))
    theta_ = cur_err_state[2]
    alpha = cur_ref_state[2]
    rot_3d_z = np.zeros((3,2))
    rot_3d_z[0,0] = np.cos(theta_+alpha)
    rot_3d_z[0,1] = 0
    rot_3d_z[1,0] = np.sin(theta_+alpha)
    rot_3d_z[1,1] = 0
    rot_3d_z[2,0] = 0
    rot_3d_z[2,1] = 1

    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        next_err_state = cur_err_state + time_step*f + (cur_ref_state - next_ref_state) + w 
    else:
        next_err_state = cur_err_state + time_step*f + (cur_ref_state - next_ref_state)
    next_theta = next_err_state[2] + next_ref_state[2]
    next_theta = fmod((next_theta + pi), 2*pi) - pi  # wrapping theta
    #next_theta = atan2(sin(next_theta), cos(next_theta))  # wrapping theta
    next_err_state[2] = next_theta - next_ref_state[2]
    return next_err_state

def gaussian_weights(mean, invcovar, x):
    exponent = np.exp(-0.5*(x-mean).T @ invcovar @ (x-mean))
    return exponent

def transition_probabilities(time_step, cur_t, cur_err_state, control):
    var_xy = 0.04
    sd_xy = np.sqrt(var_xy) 
    var_theta = 0.004
    sd_theta = np.sqrt(var_theta) 
    next_err_state = g_func(time_step, cur_t, cur_err_state, control)
    ex, ey, etheta = next_err_state
    nx = (2*(np.ceil((3*sd_xy)/P['res'])+1)).astype(np.int32)  # 97% samples around mean
    ny = nx
    ntheta = (2*(np.ceil((3*sd_theta)/THETA['res'])+1)).astype(np.int32) 
    x_vals = ex + P['res']*np.array(list(range(-int(nx/2), int(nx/2))))
    y_vals = ey + P['res']*np.array(list(range(-int(ny/2), int(ny/2))))
    theta_vals = etheta + THETA['res']*np.array(list(range(-int(ntheta/2), int(ntheta/2))))

    prob_weights = np.zeros((nx,ny,ntheta))
    mean = next_err_state
    covar = np.eye(3)
    covar[0,0] = 0.04
    covar[1,1] = 0.04
    covar[2,2] = 0.004
    invcovar = np.linalg.inv(covar)
    for i in tqdm(range(nx)):
        for j in range(ny):
            for k in range(ntheta):
                prob_weights[i, j, k] = gaussian_weights(mean, invcovar, np.array([x_vals[i], y_vals[j], theta_vals[k]]))          
    probabilities = prob_weights/np.sum(prob_weights)
    return probabilities

def IsInvalid(x):
    t, e = x
    p = e + np.array(lissajous(t))
    obstacle_centers = [[-2, -2], [1, 2]]  
    obstacle_radii = [0.5, 0.5]
    xyw_lb = np.array([-3, -3, -np.pi])
    xyw_ub = np.array([3, 3, np.pi])
    for s in range(3): # checking boundaries
        if s == 2:
            if p[s] < xyw_lb[s]:
                return True
        else:
            if p[s] <= xyw_lb[s]:
                return True
        if p[s] >= xyw_ub[s]:
            return True 
    for center, radius in zip(obstacle_centers, obstacle_radii):
        distance = np.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2)
        if distance <= radius: # collision
            return True
    return False

def stage_cost(x, u):
    if IsInvalid(x):
        return np.inf
    t, e = x
    e_next = g_func(time_step, t, e, u, noise = False)
    x_next = [(t+1)%N_t, e_next]
    if IsInvalid(x_next):
        cost = np.inf
    else:
        p_ = e[0:2]
        theta_ = e[2]
        cost = p_.T @ Q @ p_ + q*(1-np.cos(theta_))**2 + u.T @ R @ u
    return cost

def get_indices(x):
    t, e = x
    ex, ey, etheta = e
    n_ex = np.floor((ex-pmin[0])/P['res']).astype(np.int32)
    n_ey = np.floor((ey-pmin[1])/P['res']).astype(np.int32)
    n_etheta = np.floor((etheta-thetamin)/THETA['res']).astype(np.int32)
    return t, n_ex, n_ey, n_etheta

def Hamiltonian(x, u, V, gamma):
    # Based on deterministic motion model
    L = stage_cost(x, u)
    t, e = x
    e_next = g_func(time_step, t, e, u, noise = False)
    x_next = [(t+1)%N_t, e_next]
    a,b,c,d = get_indices(x_next)
    V_next = L + gamma*V[a, b, c, d]
    return V_next

def Bellman(x, pi_u, V, gamma):
    return Hamiltonian(x, pi_u, V, gamma)

def GPI(V0, numEvalIter):
    #transition_probabilities(0.5, 0, np.array([0, 0, 0]), np.array([0.5, 0.5]))
    V = V0
    Policy = np.zeros((N_t, nx, ny, ntheta, 2))
    for _ in tqdm(range(100)):
        dumm=6
        # Policy improvement: Given Value function, find optimal policy
        for i in tqdm(range(N_t)):
            for j in range(nx):
                for k in range(ny):
                    for l in range(ntheta):
                        x = [i, np.array([x_values[j], y_values[k], theta_values[l]])]
                        store_hamilt = np.zeros((n_v, n_omega))
                        for r in range(n_v):
                            for s in range(n_omega):
                                u = np.array([Vel['v'][r], OMEGA['omega'][s]])
                                store_hamilt[r, s] = Hamiltonian(x, u, V, gamma)
                        #min_V = np.min(store_hamilt)
                        min_index = np.unravel_index(np.argmin(store_hamilt), store_hamilt.shape)
                        if np.sum(min_index) != 0:
                            stap=2
                        Policy[i,j,k,l,:] = np.array([Vel['v'][min_index[0]], OMEGA['omega'][min_index[1]]])

        # Policy evaluation: Given policy, evaluate value function
        for it in range(numEvalIter):
            for i in range(N_t):
                for j in range(nx):
                    for k in range(ny):
                        for l in range(ntheta):
                            x = [i, np.array([x_values[j], y_values[k], theta_values[l]])]
                            pi_u = Policy[i,j,k,l,:]
                            V[i,j,k,l] = Bellman(x, pi_u, V, gamma) # Gauss-Seidel type evaluation
    bh=4

############################################
# RUN GPI
nx = P['size'][0]
ny = P['size'][1]
ntheta = THETA['size']
x_values = P['x']
y_values = P['y']
theta_values = THETA['theta']
n_v = Vel['size']
n_omega = OMEGA['size'] 

V0 = np.zeros((N_t, nx, ny, ntheta))



GPI(V0, 5)