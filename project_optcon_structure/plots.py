#
# OPTCON-RL 2026/2027
# Plots required by the project
#

import numpy as np
import matplotlib.pyplot as plt

import cost as cst
import armijo
import armijo_merit
import algo_newton_optcon
import algo_sqp_optcon


def plot_trajectory(xx, uu, xx_ref, uu_ref, dt, title, labels=('optimal', 'desired')):
  """
    States and inputs of a trajectory against a reference curve, one subplot per signal

    Example
      fig = plots.plot_trajectory(xx, uu, xx_ref, uu_ref, dyn.dt, 'Optimal trajectory and desired curve')
      fig = plots.plot_trajectory(xx, uu, xx_gen, uu_gen, dyn.dt, 'LQR tracking', labels=('LQR', 'generated'))
  """
  # TODO 12: compute fig: figure of the states and inputs of (xx, uu) against (xx_ref, uu_ref), one subplot per signal
  raise NotImplementedError("TODO 12: compute fig: figure of the states and inputs of (xx, uu) against (xx_ref, uu_ref), one subplot per signal")

  return fig


def plot_iterations(log, xx_ref, uu_ref, dt, title, n_show=4):
  """
    Desired curve, a few intermediate iterates and the last one

    Example
      fig = plots.plot_iterations(log, xx_ref, uu_ref, dyn.dt, 'Intermediate trajectories')
  """
  # TODO 13: compute fig: figure of the desired curve and a few iterates of log['xx_iters'], log['uu_iters'], one subplot per signal
  raise NotImplementedError("TODO 13: compute fig: figure of the desired curve and a few iterates of log['xx_iters'], log['uu_iters'], one subplot per signal")

  return fig


def plot_convergence(log, title):
  """
    Norm of the descent direction and cost along the iterations (semi-logarithmic scale),
    plus the constraint violation when available (SQP)

    Example
      fig = plots.plot_convergence(log, "Newton's method")       # log returned by solve
  """
  # TODO 14: compute fig: figure of log['descent_norm'], log['cost'] (and log['constraint_norm'], if present) along the iterations, semilog
  raise NotImplementedError("TODO 14: compute fig: figure of log['descent_norm'], log['cost'] (and log['constraint_norm'], if present) along the iterations, semilog")

  return fig


def plot_armijo(dyn, log, kk, xx_ref, uu_ref, prm):
  """
    Armijo descent direction plot at iteration kk of Newton's method

    Example
      # Armijo plot of iteration 0; the returned quantities are the ones given to select_stepsize
      fig, cost_fn, JJ, descent_arm = plots.plot_armijo(dyn, log, 0, xx_ref, uu_ref, prm)
  """
  QQT = cst.terminal_weight(dyn, xx_ref[:,-1], uu_ref[:,-1], prm.QQt, prm.RRt)
  xx, uu = log['xx_iters'][kk], log['uu_iters'][kk]

  KK, sigma, dx, du, descent_arm, JJ = algo_newton_optcon.descent_direction(dyn, xx, uu, xx_ref, uu_ref, prm, QQT)

  def cost_fn(stepsize):
    xx_temp, uu_temp = algo_newton_optcon.closed_loop_update(dyn, xx, uu, KK, sigma, stepsize)
    return cst.trajectory_cost(xx_temp, uu_temp, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT, getattr(prm, 'input_bounds', None))

  armijo.select_stepsize(cost_fn, JJ, descent_arm, prm.armijo_stepsize0, prm.armijo_c, prm.armijo_beta,
                         prm.armijo_max_iters, plot=True)
  fig = plt.gcf()
  fig.axes[0].set_title(f'Armijo, iteration {kk}')

  return fig, cost_fn, JJ, descent_arm


def plot_armijo_merit(dyn, log, kk, xx_ref, uu_ref, prm):
  """
    Armijo descent direction plot of iteration kk of the SQP method: merit function
    along the direction, its tangent at zero and the Armijo line, with the tested stepsizes

    Return
      - fig, merit_fn, M0, DM0: the figure and the quantities given to armijo_merit.select_stepsize

    Example
      fig, merit_fn, M0, DM0 = plots.plot_armijo_merit(dyn, log, 0, xx_ref, uu_ref, prm)   # log of algo_sqp_optcon.solve
  """
  QQT = getattr(prm, 'QQT', None)
  if QQT is None:
    QQT = cst.terminal_weight(dyn, xx_ref[:,-1], uu_ref[:,-1], prm.QQt, prm.RRt)
  xx, uu = log['xx_iters'][kk], log['uu_iters'][kk]

  dx, du, lmbd_qp, ss, M0, DM0, JJ = algo_sqp_optcon.descent_direction(dyn, xx, uu, xx_ref, uu_ref, prm, QQT)
  merit_fn = algo_sqp_optcon.merit_function(dyn, xx, uu, dx, du, xx_ref, uu_ref, prm, QQT, ss)

  armijo_merit.select_stepsize(merit_fn, M0, DM0, prm.armijo_stepsize0, prm.armijo_c, prm.armijo_beta,
                               prm.armijo_max_iters, plot=True)
  fig = plt.gcf()
  fig.axes[0].set_title(f'Armijo on the merit function, iteration {kk}')

  return fig, merit_fn, M0, DM0


def plot_tracking_error(runs, xx_des, dt, title, target='gen'):
  """
    Norm of the tracking error for a list of runs (label, xx, uu), with respect to
    the curve xx_des (ns, TT); target is its name in the label of the axis ('gen', 'ref')

    Example
      runs = [('perturbation x 1.0', xx, uu)]                   # one entry for each initial condition
      fig = plots.plot_tracking_error(runs, xx_gen, dyn.dt, 'LQR: tracking error')
      fig = plots.plot_tracking_error(runs, xx_ref, dyn.dt, 'MPC: tracking error', target='ref')
  """
  # TODO 21: compute fig: figure of the norm of the tracking error xx - xx_des of each run along time, semilog
  raise NotImplementedError("TODO 21: compute fig: figure of the norm of the tracking error xx - xx_des of each run along time, semilog")

  return fig


def plot_inputs(runs, dt, title, u_min=None, u_max=None):
  """
    Inputs of a list of runs (label, xx, uu), one panel for each input, with the box
    constraint u_min <= u <= u_max (dashed lines, region outside the box shaded) if given

    Example
      runs = [('perturbation x 1.0', xx, uu)]                   # e.g., the runs of the MPC
      fig = plots.plot_inputs(runs, dyn.dt, 'MPC: inputs', prm.mpc_u_min, prm.mpc_u_max)
  """
  # TODO 25: compute fig: figure of the inputs of the runs along time, with the box [u_min, u_max] if given
  raise NotImplementedError("TODO 25: compute fig: figure of the inputs of the runs along time, with the box [u_min, u_max] if given")

  return fig
