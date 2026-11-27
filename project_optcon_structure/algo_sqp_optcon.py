#
# OPTCON-RL 2026/2027
# SQP method for Optimal Control
# Merit function M1 = J + s*||h||_1 and Armijo stepsize selection on it
#

from asyncio import log

import numpy as np

import cost as cst
import armijo_merit
from solver_LQR import ltv_LQR
from algo_newton_optcon import linearize


def dynamics_defect(dyn, xx, uu):
  """
    Violation of the dynamics constraint along the curve (xx, uu)

      h_t = f(x_t, u_t) - x_{t+1},   t = 0,...,TT-2

    Return
      - hh (ns, TT-1)

    Example
      hh = algo_sqp_optcon.dynamics_defect(dyn, xx, uu)
      constraint_norm = np.linalg.norm(hh)      # zero if (xx, uu) is a trajectory
  """
  ns, TT = xx.shape

  hh = np.zeros((ns, TT-1))
  for tt in range(TT-1):
    hh[:,tt] = dyn.dynamics(xx[:,tt], uu[:,tt], jac=False)[0] - xx[:,tt+1]

  return hh


def descent_direction(dyn, xx, uu, xx_ref, uu_ref, prm, QQT):
  """
    SQP direction at the iterate (xx, uu): solution of the affine LQ problem with the
    dynamics linearized about (xx, uu) and their defect as affine term


    Return
      - dx (ns, TT), du (ni, TT): direction
      - lmbd_qp (ns, TT): multipliers of the LQ problem
      - ss: penalty parameter of the merit function M1(z; s) = J(z) + s*||h(z)||_1
      - M0: merit function at (xx, uu)
      - DM0: directional derivative of the merit function along (dx, du)
      - JJ: cost of (xx, uu)

    Example
      dx, du, lmbd_qp, ss, M0, DM0, JJ = algo_sqp_optcon.descent_direction(dyn, xx, uu, xx_ref, uu_ref, prm, QQT)
  """
  ns, TT = xx.shape

  #P# h_t(x, u) = f_t(x_t, u_t) - x_{t+1}, vedi pacco 8 - SQP for optimal control e pacco 3 - Dynamics as equality constraints
  hh = dynamics_defect(dyn, xx, uu) 


  AA, BB, QQ, RR, SS, qq, rr, qqT, JJ = linearize(dyn, xx, uu, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT,
                                                  bounds=getattr(prm, 'input_bounds', None))

  # TODO 7: compute cc (ns, TT): affine term of the linearized dynamics, from the defect hh
  # raise NotImplementedError("TODO 7: compute cc (ns, TT): affine term of the linearized dynamics, from the defect hh")
  #P# ct := ft(xt, ut) − xt+1, vedi pacco 8 - SQP for optimal control
  x0 = np.zeros((ns,))        # the initial state is fixed
  cc = np.zeros((ns, TT))
  cc[:, :-1] = hh #P# cc e hh sono di base la stessa cosa stando alla teoria, semplicemente cambia la dimensione 


  _, _, dx, du, lmbd_qp = ltv_LQR(AA, BB, QQ, RR, SS, QQT, qq, rr, qqT, cc, x0)

  # Merit function M1(z; s) = J(z) + s*||h(z)||_1

  # TODO 8: compute ss, M0, DM0: weight of the merit function, M1 = J + ss ||h||_1 at the iterate, and its directional derivative
  # raise NotImplementedError("TODO 8: compute ss, M0, DM0: weight of the merit function, M1 = J + ss ||h||_1 at the iterate, and its directional derivative")
  #P# si veda pacco 2 - Non linear Optimization, il nostro cost.py e il file SQP.py del lab 2
  ss = 100*np.ceil(np.linalg.norm([lmbd_qp], np.inf))  #P# ss: penalty of the merit function
  M0 = JJ + ss * np.linalg.norm([hh], 1)
  # DM0 = dlk.T@deltaz - ss*np.linalg.norm([hh], 1)
  # Si veda pacco 8 - SQP for optimal control 
  grad_dot_d = ( # $$\sum_{t=0}^{T-1} \begin{bmatrix} q_t \\ r_t \end{bmatrix}^\top \begin{bmatrix} \Delta x_t \\ \Delta u_t \end{bmatrix} + q_T^\top \Delta x_T 
      np.sum(qq[:, :-1] * dx[:, :-1]) # q_t^T*\Delta x_t
      + np.sum(rr[:, :-1] * du[:, :-1]) # r_t^T*\Delta u_t
      + np.dot(qqT.ravel(), dx[:, -1]) # q_T^T*\Delta x_T
  )

  DM0 = grad_dot_d - ss * np.linalg.norm([hh], 1)

  return dx, du, lmbd_qp, ss, M0, DM0, JJ




def merit_function(dyn, xx, uu, dx, du, xx_ref, uu_ref, prm, QQT, ss):
  """
    Merit function M1(z; s) = J(z) + s*||h(z)||_1 along the direction (dx, du),
    as a function of the stepsize only

    Return
      - merit_fn: merit_fn(stepsize) = M1((xx, uu) + stepsize * (dx, du); ss)

    Example
      merit_fn = algo_sqp_optcon.merit_function(dyn, xx, uu, dx, du, xx_ref, uu_ref, prm, QQT, ss)
      M0 = merit_fn(0.0)
  """
  def merit_fn(stepsize):
    xx_temp = xx + stepsize * dx
    uu_temp = uu + stepsize * du
    JJ_temp = cst.trajectory_cost(xx_temp, uu_temp, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT, getattr(prm, 'input_bounds', None))
    return JJ_temp + ss * np.sum(np.abs(dynamics_defect(dyn, xx_temp, uu_temp)))

  return merit_fn


def solve(dyn, xx_ref, uu_ref, xx_init, uu_init, prm):
  """
    SQP method for optimal control

    Args
      - dyn: dynamics module
      - xx_ref (ns, TT), uu_ref (ni, TT): reference curve
      - xx_init (ns, TT), uu_init (ni, TT): initial guess. It does not need to be a
        trajectory; xx_init[:,0] is the (fixed) initial state
      - prm: parameters (QQt, RRt, max_iters, term_cond, armijo_*, show_plots)
          if prm has the (optional) entry QQT, it is the terminal weight;
          otherwise the terminal weight is cost.terminal_weight at the end of the reference;
          if prm has the (optional) entry input_bounds = (u_min, u_max, eps, delta), the barrier
          of the input box constraint is added to the cost (cost.input_barrier; set by tracking_MPC)

    Return
      - xx (ns, TT), uu (ni, TT): last iterate
      - log: dictionary with, for each iteration,
          'cost', 'descent_norm', 'constraint_norm', 'descent_slope', 'stepsize',
          'xx_iters' (n_iters, ns, TT), 'uu_iters' (n_iters, ni, TT)

    Example
      # initial guess: the reference curve, which is not a trajectory
      xx, uu, log = algo_sqp_optcon.solve(dyn, xx_ref, uu_ref, xx_ref, uu_ref, prm)
      print('iterations:', len(log['cost']) - 1, ' constraint violation:', log['constraint_norm'][-1])
  """
  ns, TT = xx_ref.shape

  # Terminal weight: the one given in the parameters, if any
  QQT = getattr(prm, 'QQT', None)
  if QQT is None:
    QQT = cst.terminal_weight(dyn, xx_ref[:,-1], uu_ref[:,-1], prm.QQt, prm.RRt)

  xx = np.array(xx_init, dtype=float)
  uu = np.array(uu_init, dtype=float)

  log = {'cost': [], 'descent_norm': [], 'constraint_norm': [], 'descent_slope': [], 'stepsize': [],
         'xx_iters': [], 'uu_iters': []}


  print('-*-*-*-*-*-')

  for kk in range(prm.max_iters + 1):

    # Step 1: descent direction = affine LQR on the linearized problem

    try:
      if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(uu))):
        raise np.linalg.LinAlgError('the iterate is not finite')


      # TODO 9: compute dx, du, lmbd_qp, ss, M0, DM0, JJ (descent_direction at the current iterate) and descent_norm, the norm of (dx, du)
      #raise NotImplementedError("TODO 9: compute dx, du, lmbd_qp, ss, M0, DM0, JJ (descent_direction at the current iterate) and descent_norm, the norm of (dx, du)")
      

      #S# dx, du, lmbd_qp, ss, M0, DM0, JJ --> presi direttamente dal "Return" dalla funzione descent_direction
      
      dx, du, lmbd_qp, ss, M0, DM0, JJ = descent_direction(dyn, xx, uu, xx_ref, uu_ref, prm, QQT)

      descent_norm = np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(du)**2) 

      constraint_norm = np.linalg.norm(dynamics_defect(dyn, xx, uu))

    except np.linalg.LinAlgError:
      # the iterates diverged: the LQ problem cannot be solved
      DM0, JJ, descent_norm, constraint_norm = np.nan, np.inf, np.nan, np.nan

    log['cost'].append(JJ)
    log['descent_norm'].append(descent_norm)
    log['constraint_norm'].append(constraint_norm)
    log['descent_slope'].append(DM0)
    log['xx_iters'].append(xx.copy())
    log['uu_iters'].append(uu.copy())

    print(f'Iter = {kk}\t Descent = {descent_norm:.3e}\t Constr = {constraint_norm:.3e}\t Cost = {JJ:.3e}')

    if not np.isfinite(JJ) or not np.isfinite(descent_norm):
      print('The cost is not finite. Terminating.')
      break

    if descent_norm <= prm.term_cond and constraint_norm <= prm.term_cond:
      break

    if kk == prm.max_iters:
      print('Maximum number of iterations reached.')
      break

    # Step 2: Armijo on the merit function

    merit_fn = merit_function(dyn, xx, uu, dx, du, xx_ref, uu_ref, prm, QQT, ss)

    # TODO 10: compute stepsize: Armijo rule on the merit function (armijo_merit.select_stepsize with merit_fn, M0, DM0)
    #raise NotImplementedError("TODO 10: compute stepsize: Armijo rule on the merit function (armijo_merit.select_stepsize with merit_fn, M0, DM0)")

    stepsize = armijo_merit.select_stepsize(merit_fn, M0, DM0, prm.armijo_stepsize0, prm.armijo_c, prm.armijo_beta, prm.armijo_max_iters, plot=False)    
   
    log['stepsize'].append(stepsize)

    # Step 3: update (possibly violating the dynamics)

    xx = xx + stepsize * dx
    uu = uu + stepsize * du

  for key in log:
    log[key] = np.array(log[key])

  return xx, uu, log
