#
# OPTCON-RL 2026/2027
# Task 4 - Tracking of the reference curve via nonlinear MPC
#

import numpy as np
import matplotlib.pyplot as plt

import parameters
import dynamics as dyn
import common
import plots
import tracking_LQR
import tracking_MPC
import reference_curve


def run(prm):
  """
    Tracking of the smooth reference curve of Task 2 (reference_curve.smooth_reference) from a
    perturbed initial condition, with the MPC (prm.mpc_solver: 'newton' or 'sqp', horizon
    prm.mpc_T_pred); comparison with the LQR of Task 3, which tracks the generated trajectory
    instead. All the errors are measured with respect to the reference curve.

    Saves x_mpc.npy, u_mpc.npy in prm.save_dir: the MPC closed loop

    Return
      - dictionary with 'comparison': list of (label, xx, uu)
  """
  TT = int(round(prm.tf / dyn.dt))
  xx_ref, uu_ref = reference_curve.smooth_reference(TT, prm)

  x0 = xx_ref[:,0] + prm.tracking_x0_perturbation

  xx_gen, uu_gen = common.load_generated_trajectory(prm)
  if xx_gen.shape != xx_ref.shape:
    raise ValueError(f"the generated trajectory has {xx_gen.shape[1]} samples, the reference curve {xx_ref.shape[1]}: "
                     "run main_task2.py again with the current parameters")

  # TODO 23: compute xx_mpc, uu_mpc (MPC from x0, tracking_MPC.track) and xx_lqr, uu_lqr (LQR of Task 3 from the same x0)
  raise NotImplementedError("TODO 23: compute xx_mpc, uu_mpc (MPC from x0, tracking_MPC.track) and xx_lqr, uu_lqr (LQR of Task 3 from the same x0)")

  common.save_array(prm, 'x_mpc.npy', xx_mpc)
  common.save_array(prm, 'u_mpc.npy', uu_mpc)

  comparison = [(f'MPC ({prm.mpc_solver}), horizon {prm.mpc_T_pred}', xx_mpc, uu_mpc),
                ('LQR (generated trajectory)', xx_lqr, uu_lqr)]

  figures = {}
  figures['task4_trajectory'] = plots.plot_trajectory(xx_mpc, uu_mpc, xx_ref, uu_ref, dyn.dt,
                                                    'MPC tracking', labels=('MPC', 'reference'))
  u_min, u_max = getattr(prm, 'mpc_u_min', None), getattr(prm, 'mpc_u_max', None)
  bounds_title = 'MPC: inputs and input bounds' if u_min is not None else 'MPC: inputs (no input bounds set: mpc_u_min = None)'
  figures['task4_inputs'] = plots.plot_inputs([comparison[0]], dyn.dt, bounds_title, u_min, u_max)
  figures['task4_comparison'] = plots.plot_tracking_error(comparison, xx_ref, dyn.dt,
                                                        'Distance from the reference curve: MPC and LQR',
                                                        target='ref')

  common.finalize(prm, figures)

  return {'comparison': comparison}


if __name__ == '__main__':
  run(parameters)
