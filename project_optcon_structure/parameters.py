#
# OPTCON-RL 2026/2027
# Parameters File
#
# All the choices of the project are collected here.
# The model (dimensions, sampling time, physical parameters) is in dynamics.py.
#

import numpy as np

#######################################
# Reference curve
#######################################

tf = 10.0                           # final time in seconds

reference_angle_initial_deg = 0.0
reference_angle_final_deg = 30.0

transition_fraction = 0.2           # fraction of the horizon used by the smooth transition

#######################################
# Optimal trajectory generation
#######################################

# Quadratic cost
QQt = np.diag([10.0, 1.0])
RRt = np.eye(1)

# Method for the trajectory generation: 'newton' or 'sqp'
solver = 'newton'

# Newton's method / SQP
max_iters = 100
term_cond = 1e-6

# Armijo parameters
armijo_stepsize0 = 1.0
armijo_c = 0.5
armijo_beta = 0.7
armijo_max_iters = 30

#######################################
# Trajectory tracking
#######################################

QQt_reg = np.diag([1e2, 1e1])
RRt_reg = np.eye(1)
QQT_reg = np.diag([1e2, 1e1])

tracking_x0_perturbation = np.array([0.3, 0.0])   # added to the initial state of the trajectory

# MPC
mpc_T_pred = 50               # prediction horizon (samples)
mpc_solver = 'newton'         # method used to solve the problem at each time: 'newton' or 'sqp'
mpc_max_iters = 3             # maximum number of iterations of the method at each time

# Box constraint on the input of the MPC, through a relaxed log barrier (cost.input_barrier)
# None: no constraint. The input of the reference curve must be strictly inside the box.
mpc_u_min = np.array([-0.3])  # None: no constraint; one entry per input
mpc_u_max = np.array([1.0])
mpc_barrier_eps = 0.1         # weight of the barrier
mpc_barrier_delta = 1.3e-3    # below this distance from a bound the log is continued by a quadratic of
                              # curvature eps/delta^2: keep it small (about 1e-3 of the width of the box)
                              # so that the continuation is stiff with respect to the tracking weights

#######################################
# Visualization
#######################################

show_plots = True
show_descent_plot = False           # True: Armijo plot at every iteration
save_dir = 'results'                # folder for the generated trajectory and the figures
