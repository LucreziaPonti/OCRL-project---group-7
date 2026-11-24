#
# OPTCON-RL 2026/2027
# Newton's method for Optimal Control (closed-loop version)
# Regularized version: Hessians of the cost only (Gauss-Newton)
#

import numpy as np

import cost as cst
import armijo
from solver_LQR import ltv_LQR


def linearize(dyn, xx, uu, xx_ref, uu_ref, QQt, RRt, QQT, bounds=None):
  """
    Linearization of the dynamics and quadratic model of the cost
    about the state-input curve (xx, uu)

    Args
      - dyn: dynamics module
      - xx (ns, TT), uu (ni, TT): current curve
      - xx_ref (ns, TT), uu_ref (ni, TT): reference curve
      - QQt, RRt, QQT: cost weights
      - bounds: None (default), or the bounds of the input barrier (see cost.stagecost)

    Return
      - AA (ns, ns, TT), BB (ns, ni, TT): A_t = dfx.T, B_t = dfu.T
      - QQ (ns, ns, TT), RR (ni, ni, TT), SS (ni, ns, TT): Hessians of the cost
      - qq (ns, TT), rr (ni, TT), qqT (ns,): gradients of the cost
      - JJ: cost of the curve

    Example
      AA, BB, QQ, RR, SS, qq, rr, qqT, JJ = algo_newton_optcon.linearize(dyn, xx, uu, xx_ref, uu_ref,
                                                                    prm.QQt, prm.RRt, QQT)
  """
  ns, TT = xx.shape
  ni = uu.shape[0]

  AA = np.zeros((ns, ns, TT))
  BB = np.zeros((ns, ni, TT))
  QQ = np.zeros((ns, ns, TT))
  RR = np.zeros((ni, ni, TT))
  SS = np.zeros((ni, ns, TT))
  qq = np.zeros((ns, TT))
  rr = np.zeros((ni, TT))

  JJ = 0

  for tt in range(TT-1):

    # TODO 2: compute AA[:,:,tt], BB[:,:,tt], QQ[:,:,tt], RR[:,:,tt], SS[:,:,tt], qq[:,tt], rr[:,tt] and add the stage cost to JJ: linearization of the dynamics and quadratic model of the cost at time tt
    raise NotImplementedError("TODO 2: compute AA[:,:,tt], BB[:,:,tt], QQ[:,:,tt], RR[:,:,tt], SS[:,:,tt], qq[:,tt], rr[:,tt] and add the stage cost to JJ: linearization of the dynamics and quadratic model of the cost at time tt")

  lT, qqT, _ = cst.termcost(xx[:,-1], xx_ref[:,-1], QQT)
  JJ += lT


  return AA, BB, QQ, RR, SS, qq, rr, qqT, JJ


def descent_direction(dyn, xx, uu, xx_ref, uu_ref, prm, QQT):
  """
    Descent direction at the trajectory (xx, uu) as solution of an affine LQ problem


    Return
      - KK (ni, ns, TT), sigma (ni, TT): feedback gains and feedforward terms
      - dx (ns, TT), du (ni, TT): descent direction
      - descent_arm: Armijo descent direction, i.e., directional derivative of the cost along (dx, du)
      - JJ: cost of the trajectory (xx, uu)

    Example
      KK, sigma, dx, du, descent_arm, JJ = algo_newton_optcon.descent_direction(dyn, xx, uu, xx_ref, uu_ref,
                                                                          prm, QQT)
      descent_norm = np.sqrt(np.linalg.norm(dx)**2 + np.linalg.norm(du)**2)
  """
  ns, TT = xx.shape


  AA, BB, QQ, RR, SS, qq, rr, qqT, JJ = linearize(dyn, xx, uu, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT,
                                                  bounds=getattr(prm, 'input_bounds', None))

  cc = np.zeros((ns, TT))     # (xx, uu) is a trajectory: no affine term in the dynamics
  x0 = np.zeros((ns,))        # the initial condition is fixed

  KK, sigma, dx, du, _ = ltv_LQR(AA, BB, QQ, RR, SS, QQT, qq, rr, qqT, cc, x0)

  descent_arm = np.sum(qq[:,:-1] * dx[:,:-1]) + np.sum(rr[:,:-1] * du[:,:-1]) + qqT @ dx[:,-1]

  return KK, sigma, dx, du, descent_arm, JJ




def closed_loop_update(dyn, xx, uu, KK, sigma, stepsize):
  """
    Closed-loop update of the trajectory

      u_t^{k+1} = u_t^k + stepsize * sigma_t + K_t (x_t^{k+1} - x_t^k)
      x_{t+1}^{k+1} = f(x_t^{k+1}, u_t^{k+1})

    Return
      - xx_new (ns, TT), uu_new (ni, TT): the new trajectory

    Example
      xx_new, uu_new = algo_newton_optcon.closed_loop_update(dyn, xx, uu, KK, sigma, stepsize)
  """
  TT = xx.shape[1]

  xx_new = np.zeros(xx.shape)
  uu_new = np.zeros(uu.shape)

  xx_new[:,0] = xx[:,0]

  for tt in range(TT-1):
    # TODO 4: compute uu_new[:,tt]: closed-loop update of the input, uu + stepsize*sigma + KK (xx_new - xx)
    raise NotImplementedError("TODO 4: compute uu_new[:,tt]: closed-loop update of the input, uu + stepsize*sigma + KK (xx_new - xx)")
    xx_new[:,tt+1] = dyn.dynamics(xx_new[:,tt], uu_new[:,tt], jac=False)[0]

  uu_new[:,-1] = uu_new[:,-2]     # not used, for plots only

  return xx_new, uu_new


def solve(dyn, xx_ref, uu_ref, xx_init, uu_init, prm):
  """
    Newton's method for optimal control

    Args
      - dyn: dynamics module
      - xx_ref (ns, TT), uu_ref (ni, TT): reference curve
      - xx_init (ns, TT), uu_init (ni, TT): initial guess. The initial trajectory
        is the forward simulation of uu_init from xx_init[:,0]
      - prm: parameters (QQt, RRt, max_iters, term_cond, armijo_*, show_plots)
          if prm has the (optional) entry QQT, it is the terminal weight;
          otherwise the terminal weight is cost.terminal_weight at the end of the reference;
          if prm has the (optional) entry input_bounds = (u_min, u_max, eps, delta), the barrier
          of the input box constraint is added to the cost (cost.input_barrier; set by tracking_MPC)

    Return
      - xx (ns, TT), uu (ni, TT): last trajectory
      - log: dictionary with, for each iteration,
          'cost', 'descent_norm', 'descent_slope', 'stepsize',
          'xx_iters' (n_iters, ns, TT), 'uu_iters' (n_iters, ni, TT)

    Example
      # initial guess: the first equilibrium held for the whole horizon
      xx, uu, log = algo_newton_optcon.solve(dyn, xx_ref, uu_ref, xx_init, uu_init, prm)
      print('iterations:', len(log['cost']) - 1, ' final cost:', log['cost'][-1])
  """
  TT = xx_ref.shape[1]


  # Terminal weight: the one given in the parameters, if any
  QQT = getattr(prm, 'QQT', None)
  if QQT is None:
    QQT = cst.terminal_weight(dyn, xx_ref[:,-1], uu_ref[:,-1], prm.QQt, prm.RRt)

  # Initial trajectory: forward simulation of the initial input guess

  uu = np.array(uu_init, dtype=float)
  xx = np.zeros(xx_ref.shape)
  xx[:,0] = xx_init[:,0]
  for tt in range(TT-1):
    xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], jac=False)[0]

  log = {'cost': [], 'descent_norm': [], 'descent_slope': [], 'stepsize': [], 'xx_iters': [], 'uu_iters': []}

  print('-*-*-*-*-*-')

  for kk in range(prm.max_iters + 1):

    # Step 1: descent direction

    try:
      if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(uu))):
        raise np.linalg.LinAlgError('the trajectory is not finite')
      # TODO 3: compute KK, sigma, dx, du, descent_arm, JJ (descent_direction at the current trajectory) and descent_norm, the norm of (dx, du)
      raise NotImplementedError("TODO 3: compute KK, sigma, dx, du, descent_arm, JJ (descent_direction at the current trajectory) and descent_norm, the norm of (dx, du)")
    except np.linalg.LinAlgError:
      # the iterates diverged: the LQ problem cannot be solved
      descent_arm, JJ, descent_norm = np.nan, np.inf, np.nan

    log['cost'].append(JJ)
    log['descent_norm'].append(descent_norm)
    log['descent_slope'].append(descent_arm)
    log['xx_iters'].append(xx.copy())
    log['uu_iters'].append(uu.copy())

    print(f'Iter = {kk}\t Descent = {descent_norm:.3e}\t Cost = {JJ:.3e}')

    if not np.isfinite(JJ) or not np.isfinite(descent_norm):
      print('The cost is not finite. Terminating.')
      break

    if descent_norm <= prm.term_cond:
      break

    if kk == prm.max_iters:
      print('Maximum number of iterations reached.')
      break

    # Step 2: stepsize selection - Armijo

    def cost_fn(stepsize):
      xx_temp, uu_temp = closed_loop_update(dyn, xx, uu, KK, sigma, stepsize)
      return cst.trajectory_cost(xx_temp, uu_temp, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT, getattr(prm, 'input_bounds', None))

    # TODO 5: compute stepsize: Armijo rule on the cost (armijo.select_stepsize with cost_fn, JJ, descent_arm)
    raise NotImplementedError("TODO 5: compute stepsize: Armijo rule on the cost (armijo.select_stepsize with cost_fn, JJ, descent_arm)")
    log['stepsize'].append(stepsize)

    # Step 3: update the trajectory (closed loop)

    xx, uu = closed_loop_update(dyn, xx, uu, KK, sigma, stepsize)

  for key in log:
    log[key] = np.array(log[key])

  return xx, uu, log
