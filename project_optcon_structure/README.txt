OPTCON-RL 2026/2027 - Course project - starter code (second-order methods)
==========================================================================

The full pipeline of the course on a pendulum: trajectory generation with
Newton's method and with SQP, trajectory tracking with LQR and with nonlinear MPC.
Some key parts of the algorithms have been removed: they are marked

    # TODO n: ...
    raise NotImplementedError("TODO n: ...")

Requirements: python 3, numpy, scipy, matplotlib, control, and a symbolic toolbox for the
dynamics of your system (casadi or sympy).

Structure
---------
dynamics.py            the system: dimensions, sampling time, parameters and the
                       function dynamics (next state and gradients)  (system-specific)
reference_curve.py     desired curves, starting and finishing at equilibria:
                       step_reference (between two equilibria) and smooth_reference
                       (any smooth curve), same arguments and outputs  (system-specific)
parameters.py          all the choices                                (system-specific)
common.py              helpers of the main files (saving, loading, figures)
main_task1.py          Task 1: trajectory generation on the step reference, with
                       Newton's method or SQP as chosen by solver in parameters.py
main_task2.py          Task 2: the same on the smooth reference (saves the trajectory)
main_task3.py          Task 3: tracking of the trajectory of Task 2, LQR
main_task4.py          Task 4: tracking of the smooth reference curve, nonlinear MPC
main_task5.py          Task 5: animation of the system executing one of the tasks

Working for any system (they receive the dynamics module as the argument dyn):
cost.py                stage cost, terminal cost, terminal weight, input barrier (MPC)
armijo.py              stepsize selection on the cost (Newton's method)
armijo_merit.py        stepsize selection on the merit function (SQP)
solver_LQR.py          affine LQ solver
algo_newton_optcon.py  Newton's method for optimal control (closed-loop version)
algo_sqp_optcon.py     SQP for optimal control
tracking_LQR.py        trajectory tracking via LQR
tracking_MPC.py        tracking of the reference curve via nonlinear MPC
plots.py               plots

What you have to do
-------------------
1. Fill the holes:
X     TODO 1   solver_LQR.py     compute KK[:,:,tt], sigma[:,tt], PP[:,:,tt], pp[:,tt]: backward step of the augmented Riccati equation at time tt
X     TODO 2   algo_newton_optcon.py compute AA[:,:,tt], BB[:,:,tt], QQ[:,:,tt], RR[:,:,tt], SS[:,:,tt], qq[:,tt], rr[:,tt] and add the stage cost to JJ: linearization of the dynamics and quadratic model of the cost at time tt
L     TODO 3   algo_newton_optcon.py compute KK, sigma, dx, du, descent_arm, JJ (descent_direction at the current trajectory) and descent_norm, the norm of (dx, du)
L     TODO 4   algo_newton_optcon.py compute uu_new[:,tt]: closed-loop update of the input, uu + stepsize*sigma + KK (xx_new - xx)
L     TODO 5   algo_newton_optcon.py compute stepsize: Armijo rule on the cost (armijo.select_stepsize with cost_fn, JJ, descent_arm)
P     TODO 6   main_task1.py     compute xx, uu, log: Newton's method (algo_newton_optcon.solve) from the initial guess (xx_init, uu_init)
P     TODO 7   algo_sqp_optcon.py compute cc (ns, TT): affine term of the linearized dynamics, from the defect hh
P     TODO 8   algo_sqp_optcon.py compute ss, M0, DM0: weight of the merit function, M1 = J + ss ||h||_1 at the iterate, and its directional derivative
S     TODO 9   algo_sqp_optcon.py compute dx, du, lmbd_qp, ss, M0, DM0, JJ (descent_direction at the current iterate) and descent_norm, the norm of (dx, du)
S     TODO 10  algo_sqp_optcon.py compute stepsize: Armijo rule on the merit function (armijo_merit.select_stepsize with merit_fn, M0, DM0)
S     TODO 11  main_task1.py     compute xx, uu, log: SQP (algo_sqp_optcon.solve) from the reference curve as initial guess (not a trajectory)
     TODO 12  plots.py          compute fig: figure of the states and inputs of (xx, uu) against (xx_ref, uu_ref), one subplot per signal
     TODO 13  plots.py          compute fig: figure of the desired curve and a few iterates of log['xx_iters'], log['uu_iters'], one subplot per signal
     TODO 14  plots.py          compute fig: figure of log['descent_norm'], log['cost'] (and log['constraint_norm'], if present) along the iterations, semilog
     TODO 15  armijo.py         draw the Armijo plot in a new figure: cost_fn along the direction (steps, values), the line JJ + gamma*descent_arm, the Armijo line with cc, and the tested stepsizes
     TODO 16  armijo_merit.py   draw the Armijo plot in a new figure: merit_fn along the direction (steps, values), the line M0 + gamma*DM0, the Armijo line with cc, and the tested stepsizes
     TODO 17  reference_curve.py compute xx_ref (ns, TT), uu_ref (ni, TT): smooth reference curve, starting and finishing at equilibria
     TODO 18  tracking_LQR.py   compute KK (ni, ns, TT): gains of the LQR tracking controller (ltv_LQR on the linearization AA, BB)
     TODO 19  tracking_LQR.py   compute uu[:,tt]: feedback law of the LQR, from uu_gen, KK and the error xx - xx_gen at time tt
     TODO 20  main_task3.py     compute xx, uu: LQR tracking of (xx_gen, uu_gen) from x0 (tracking_LQR.track)
     TODO 21  plots.py          compute fig: figure of the norm of the tracking error xx - xx_des of each run along time, semilog
     TODO 22  tracking_MPC.py   compute xx_sol, uu_sol and uu[:,tt]: solution of the problem on the window from the measured state, and its first input, applied to the system
     TODO 23  main_task4.py     compute xx_mpc, uu_mpc (MPC from x0, tracking_MPC.track) and xx_lqr, uu_lqr (LQR of Task 3 from the same x0)
     TODO 24  cost.py           compute bb, dbu (ni,), ddbu (ni,): value, gradient and diagonal of the Hessian of the relaxed barrier
     TODO 25  plots.py          compute fig: figure of the inputs of the runs along time, with the box [u_min, u_max] if given
     TODO 26  main_task5.py     Create an animation for the system executing task 4
   The TODOs are numbered in the order in which they are needed:
     TODO 1-2    shared by Newton's method and SQP (LQ solver, linearization)
     TODO 3-6    Newton's method, and its call in main_task1.py
     TODO 7-11   SQP, and its call in main_task1.py
                 (one of the two methods is enough: choose it with solver in parameters.py)
     TODO 12-16  plots of Tasks 1 and 2 (required by the assignment): trajectory,
                 intermediate trajectories, convergence, Armijo plot on the cost
                 (TODO 15, Newton's method) or on the merit function (TODO 16, SQP)
                 -> with TODO 1-6, 12-15 (Newton) or 1-2, 7-14, 16 (SQP) Task 1 runs
     TODO 17     smooth reference curve, used by Task 2 and Task 4
     TODO 18-20  LQR tracking, and its call in main_task3.py (Task 3)
     TODO 21     plot of the tracking error, Tasks 3 and 4
     TODO 22-23  nonlinear MPC (with either method, chosen by mpc_solver), and its call
                 in main_task4.py (Task 4)
     TODO 24     barrier of the input bounds mpc_u_min, mpc_u_max of parameters.py
                 (set mpc_u_min = None to run the MPC without bounds)
     TODO 25     plot of the inputs of the MPC with the input bounds, Task 4
     TODO 26     animation of the system executing one of the tasks, Task 5
   Both methods use the Gauss-Newton approximation of the Hessian (Hessians of the
   cost only).

2. Check the result on the pendulum: run main_task1.py to main_task4.py, in order
   (Task 3 tracks the trajectory saved by Task 2).

3. Adapt the code to the system assigned to your group: dynamics.py (ns, ni, dt,
   parameters and the function dynamics, which returns xxp (ns,), dfx (ns, ns),
   dfu (ni, ns) with A = dfx.T, B = dfu.T),
   reference_curve.py and parameters.py.
   The files from cost.py to plots.py must keep working without changes.

Array conventions, as in the code of the lectures:
  xx (ns, TT), uu (ni, TT);  the last column of uu is not used

The docstring of each function describes arguments and returned values, and ends
with an example of use. Results are saved in the folder results/.
