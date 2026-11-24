#
# OPTCON-RL 2026/2027
# Tracking of the reference curve via nonlinear MPC
# At each time the optimal control problem on the prediction window is solved
# with Newton's method or with the SQP method, using the nonlinear dynamics
#

import io
import types
import contextlib

import numpy as np

import algo_newton_optcon
import algo_sqp_optcon


def shift_guess(dyn, xx_prev, uu_prev, uu_ref_win):
  """
    Initial guess for the window starting one step later: the previous solution
    shifted by one step; the samples entering the window are the simulation of
    the input of the reference curve

    Args
      - xx_prev (ns, N_prev), uu_prev (ni, N_prev): solution on the previous window
      - uu_ref_win (ni, NN): input of the reference curve on the new window

    Return
      - xx_guess (ns, NN), uu_guess (ni, NN)

    Example
      # (xx_sol, uu_sol): solution on the window [tt, tt + NN);  new window: [tt + 1, tt + 1 + NN)
      xx_guess, uu_guess = tracking_MPC.shift_guess(dyn, xx_sol, uu_sol, uu_ref[:,tt+1:tt+1+NN])
  """
  NN = uu_ref_win.shape[1]

  xx_guess = np.zeros((xx_prev.shape[0], NN))
  uu_guess = np.array(uu_ref_win, dtype=float)

  n_old = min(NN, xx_prev.shape[1] - 1)       # samples taken from the previous solution

  xx_guess[:,:n_old] = xx_prev[:,1:1+n_old]
  uu_guess[:,:n_old-1] = uu_prev[:,1:n_old]   # the last column of uu_prev is not an input

  for kk in range(n_old-1, NN-1):
    xx_guess[:,kk+1] = dyn.dynamics(xx_guess[:,kk], uu_guess[:,kk], jac=False)[0]

  return xx_guess, uu_guess


def track(dyn, xx_ref, uu_ref, x0, prm):
  """
    Tracking of the reference curve (xx_ref, uu_ref) via nonlinear MPC

    At each time t the optimal control problem on the window [t, t + mpc_T_pred)

      min  sum (x - x_ref)' QQt_reg (x - x_ref) + (u - u_ref)' RRt_reg (u - u_ref)
           + (x_end - x_ref_end)' QQT_reg (x_end - x_ref_end)
           + sum b(u)        (only with the input bounds, see below)

      subj.to  x_{tau+1} = f(x_tau, u_tau),   x_t = measured state

    is solved with algo_newton_optcon.solve or algo_sqp_optcon.solve, starting from the
    previous solution shifted by one step, and the first input is applied to
    the system. Close to the end of the trajectory the window is shortened.

    Args
      - dyn: dynamics module
      - xx_ref (ns, TT), uu_ref (ni, TT): curve to be tracked (the reference curve of Task 2:
        it does not need to be a trajectory of the system)
      - x0 (ns,): initial condition of the system
      - prm: parameters (QQt_reg, RRt_reg, QQT_reg, mpc_T_pred, mpc_solver,
        mpc_max_iters, term_cond, armijo_*)
          mpc_solver: 'newton' or 'sqp'
          mpc_max_iters: maximum number of iterations of the method at each time
          mpc_u_min, mpc_u_max (ni,) (optional, None: no constraint): box constraint on the
            input, imposed through the relaxed barrier b(u) = cost.input_barrier(u, bounds) with
            weight mpc_barrier_eps and relaxation distance mpc_barrier_delta. The input of the
            reference curve must be strictly inside the box.

    Return
      - xx (ns, TT), uu (ni, TT): closed-loop trajectory of the nonlinear system

    Example
      x0 = xx_ref[:,0] + prm.tracking_x0_perturbation
      xx, uu = tracking_MPC.track(dyn, xx_ref, uu_ref, x0, prm)
      error = np.linalg.norm(xx - xx_ref, axis=0)
  """
  ns, TT = xx_ref.shape
  ni = uu_ref.shape[0]

  if prm.mpc_solver not in ('newton', 'sqp'):
    raise ValueError(f"mpc_solver must be 'newton' or 'sqp', not {prm.mpc_solver!r}")
  solver = algo_newton_optcon if prm.mpc_solver == 'newton' else algo_sqp_optcon

  # Parameters of the problem solved at each time: the weights of the regulator
  # take the place of the ones used for the trajectory generation
  prm_win = types.SimpleNamespace(
    QQt=prm.QQt_reg, RRt=prm.RRt_reg, QQT=prm.QQT_reg,
    max_iters=prm.mpc_max_iters, term_cond=prm.term_cond,
    armijo_stepsize0=prm.armijo_stepsize0, armijo_c=prm.armijo_c, armijo_beta=prm.armijo_beta,
    armijo_max_iters=prm.armijo_max_iters, show_plots=False)

  # Box constraint on the input, through the relaxed barrier added to the stage cost (optional)
  if getattr(prm, 'mpc_u_min', None) is not None:
    u_min = np.asarray(prm.mpc_u_min, dtype=float)
    u_max = np.asarray(prm.mpc_u_max, dtype=float)
    if not (np.all(uu_ref[:,:-1] > u_min[:,None]) and np.all(uu_ref[:,:-1] < u_max[:,None])):
      raise ValueError("the input of the reference curve must be strictly inside [mpc_u_min, mpc_u_max]")
    prm_win.input_bounds = (u_min, u_max, prm.mpc_barrier_eps, prm.mpc_barrier_delta)


  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0

  xx_sol, uu_sol = None, None

  for tt in range(TT-1):

    NN = min(prm.mpc_T_pred, TT - tt)     # prediction window (shortened at the end)

    xx_win = xx_ref[:,tt:tt+NN]
    uu_win = uu_ref[:,tt:tt+NN]

    # Initial guess

    if xx_sol is None:
      xx_guess = np.zeros((ns, NN))
      uu_guess = np.array(uu_win, dtype=float)
      xx_guess[:,0] = xx[:,tt]
      for kk in range(NN-1):
        xx_guess[:,kk+1] = dyn.dynamics(xx_guess[:,kk], uu_guess[:,kk], jac=False)[0]
    else:
      xx_guess, uu_guess = shift_guess(dyn, xx_sol, uu_sol, uu_win)
      xx_guess[:,0] = xx[:,tt]            # measured state

    # Solution of the nonlinear problem on the window (the iterations are not printed)

    # TODO 22: compute xx_sol, uu_sol and uu[:,tt]: solution of the problem on the window from the measured state, and its first input, applied to the system
    raise NotImplementedError("TODO 22: compute xx_sol, uu_sol and uu[:,tt]: solution of the problem on the window from the measured state, and its first input, applied to the system")
    xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], jac=False)[0]

  uu[:,-1] = uu[:,-2]     # not used, for plots only

  return xx, uu
