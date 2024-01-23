from time import time
import numpy as np
from utils import visualize
from casadi import *

# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

# This function implements Value function (objective) for RH-CEC controller
def RH_CEC_ValFunc(gamma, Q, R, q, Q_term, q_term, e, u_mat):
    V_val = 0
    T_horiz = u_mat.shape[1]
    for i in range(T_horiz):
        p_ = e[0:2, i]
        theta_ = e[2, i]
        u = u_mat[:, i]
        V_val = V_val + (gamma**i)*(p_.T @ Q @ p_ + q*(1-cos(theta_))**2 + u.T @ R @ u)
    p_term = e[0:2, T_horiz]
    theta_term = e[2, T_horiz]
    V_val = V_val + p_term.T @ Q_term @ p_term + q_term*(1-cos(theta_term))**2
    return V_val

def print_variable_value(iteration, solution):
    variable_value = solution['e'][2]
    print(f"Iteration {iteration}: Variable Value = {variable_value}")

# This function implements RH-CEC controller
def RH_CEC_controller(design_params, cur_state, ref_state, tau, T, time_step):
    opti = Opti()
    e_tau = cur_state - ref_state
    u = opti.variable(2,T)
    e = opti.variable(3,T+1)

    gamma_, Q_, R_, q_, Q_term_, q_term_ = design_params
    gamma = opti.parameter()
    opti.set_value(gamma, gamma_)
    Q = opti.parameter(2,2)
    opti.set_value(Q, Q_)
    R = opti.parameter(2,2)
    opti.set_value(R, R_)
    q = opti.parameter()
    opti.set_value(q, q_)
    Q_term = opti.parameter(2,2)
    opti.set_value(Q_term, Q_term_)
    q_term = opti.parameter()
    opti.set_value(q_term, q_term_)

    opti.minimize(RH_CEC_ValFunc(gamma, Q, R, q, Q_term, q_term, e, u))

    xyw_lb = np.array([-3, -3, -np.pi])
    xyw_ub = np.array([3, 3, np.pi])
    obstacle_centers = [[-2, -2], [1, 2]]  
    obstacle_radii = [0.5, 0.5] 

    u_lb = np.array([0, -1])
    u_ub = np.array([1, 1])

    opti.subject_to(e[:, 0] == e_tau)
    for i in range(T):
        opti.subject_to(e[:, i+1] == car_next_error_state(time_step=time_step, cur_t=tau+i, cur_err_state=e[:,i], control=u[:,i], noise=False)) # motion model
        opti.subject_to(u[:,i] >= u_lb) # bounds on control inputs
        opti.subject_to(u[:,i] <= u_ub)
        ref_st = np.array(lissajous(tau+i+1))
        cur_pos = e[:,i+1] + np.array(lissajous(tau+i+1))
        for s in range(2):
            opti.subject_to(e[s, i+1] > xyw_lb[s]-ref_st[s]) 
            opti.subject_to(e[s, i+1] < xyw_ub[s]-ref_st[s])
        for center, radius in zip(obstacle_centers, obstacle_radii):
            distance = sqrt((cur_pos[0] - center[0])**2 + (cur_pos[1] - center[1])**2)
            opti.subject_to(distance > radius) # avoid collision
    
    opti.solver('ipopt')
    try:
        sol = opti.solve()
    except Exception as e:
        print("Solver failed with error:", e)
        exit

    u_opt = sol.value(u)
    e_opt = sol.value(e)
    return u_opt, e_opt

# This function implements the car state error dynamics
def car_next_error_state(time_step, cur_t, cur_err_state, control, noise = False):
    cur_ref_state = np.array(lissajous(cur_t))
    next_ref_state = np.array(lissajous(cur_t+1))
    theta_ = cur_err_state[2]
    alpha = cur_ref_state[2]
    rot_3d_z = MX(3,2)
    rot_3d_z[0,0] = cos(theta_+alpha)
    rot_3d_z[0,1] = 0
    rot_3d_z[1,0] = sin(theta_+alpha)
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
    
# This function implements the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        #control = simple_controller(cur_state, cur_ref)
        gamma = 1
        if cur_iter < 30:
            Q = np.array([[1, 0], [0, 1]])
            q = 2
        else:
            Q = np.array([[2, 1], [1, 2]])
            q = 2
        R = np.array([[1, 0], [0, 1]])
        
        Q_term = np.array([[10, 2], [2, 10]])
        q_term = 5
        design_params = [gamma, Q, R, q, Q_term, q_term]
        controls, err_states = RH_CEC_controller(design_params=design_params, cur_state=cur_state, ref_state=cur_ref, tau=cur_iter, T=9, time_step=time_step)
        control = controls[:, 0]
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state  
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

