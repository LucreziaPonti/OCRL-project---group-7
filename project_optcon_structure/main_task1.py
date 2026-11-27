#
# OPTCON-RL 2026/2027
# Task 1 - Trajectory generation (I): step reference, Newton's method or SQP
#

import numpy as np
import matplotlib.pyplot as plt

import parameters
import dynamics as dyn
import common
import plots
import reference_curve
import algo_newton_optcon
import algo_sqp_optcon


def optimal_transition(prm, xx_ref, uu_ref, task):
  """
    Optimal transition along the reference curve (xx_ref, uu_ref), with the method chosen by prm.solver:
      'newton'  Newton's method (closed-loop version), from a trajectory: the first
                point of the reference curve held for the whole horizon
      'sqp'     SQP with the merit function, from an initial guess that is not a
                trajectory: the reference curve
    The figures are saved with the prefix task (e.g. 'task1').

    Return
      - xx (ns, TT), uu (ni, TT), log

    Example
      xx, uu, log = main_task1.optimal_transition(prm, xx_ref, uu_ref, 'task1')
  """
  TT = xx_ref.shape[1]

  if prm.solver == 'newton':
    xx_init = np.repeat(xx_ref[:,[0]], TT, axis=1)
    uu_init = np.repeat(uu_ref[:,[0]], TT, axis=1)
    # TODO 6: compute xx, uu, log: Newton's method (algo_newton_optcon.solve) from the initial guess (xx_init, uu_init)
    #raise NotImplementedError("TODO 6: compute xx, uu, log: Newton's method (algo_newton_optcon.solve) from the initial guess (xx_init, uu_init)")
    title = "Newton's method"
    xx, uu, log = solve(dyn, xx_ref, uu_ref, xx_init, uu_init, prm) 
  elif prm.solver == 'sqp':
    # TODO 11: compute xx, uu, log: SQP (algo_sqp_optcon.solve) from the reference curve as initial guess (not a trajectory)
    #raise NotImplementedError("TODO 11: compute xx, uu, log: SQP (algo_sqp_optcon.solve) from the reference curve as initial guess (not a trajectory)")
    title = 'SQP'
  else:
    raise ValueError(f"solver must be 'newton' or 'sqp', not {prm.solver!r}")

  print(f"{title}: {len(log['cost']) - 1} iterations, cost {log['cost'][-1]:.6e}")

  figures = {}
  figures[f'{task}_trajectory'] = plots.plot_trajectory(xx, uu, xx_ref, uu_ref, dyn.dt, f'{title}: optimal trajectory and desired curve')
  figures[f'{task}_iterations'] = plots.plot_iterations(log, xx_ref, uu_ref, dyn.dt, f'{title}: intermediate trajectories')
  figures[f'{task}_convergence'] = plots.plot_convergence(log, title)       # with SQP, also the constraint violation

  # Armijo plots at a few initial and final iterations: on the cost (Newton's method)
  # or on the merit function (SQP)

  n_iters = len(log['cost'])
  for kk in sorted(set([0, 1, max(n_iters - 3, 0), max(n_iters - 2, 0)])):
    if prm.solver == 'newton':
      figures[f'{task}_armijo_{kk}'] = plots.plot_armijo(dyn, log, kk, xx_ref, uu_ref, prm)[0]
    else:
      figures[f'{task}_armijo_{kk}'] = plots.plot_armijo_merit(dyn, log, kk, xx_ref, uu_ref, prm)[0]

  common.finalize(prm, figures)

  return xx, uu, log


def run(prm):
  """
    Optimal transition between two equilibria along the step reference curve
    (reference_curve.step_reference), with the method chosen by prm.solver

    Saves x_gen_step.npy, u_gen_step.npy in prm.save_dir

    Return
      - dictionary with 'xx', 'uu', 'log', 'xx_ref', 'uu_ref'
  """
  TT = int(round(prm.tf / dyn.dt))
  xx_ref, uu_ref = reference_curve.step_reference(TT, prm)

  xx, uu, log = optimal_transition(prm, xx_ref, uu_ref, 'task1')

  common.save_array(prm, 'x_gen_step.npy', xx)
  common.save_array(prm, 'u_gen_step.npy', uu)

  return {'xx': xx, 'uu': uu, 'log': log, 'xx_ref': xx_ref, 'uu_ref': uu_ref}


if __name__ == '__main__':
  run(parameters)
