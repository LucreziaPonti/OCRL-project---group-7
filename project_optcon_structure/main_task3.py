#
# OPTCON-RL 2026/2027
# Task 3 - Trajectory tracking via LQR
#

import numpy as np
import matplotlib.pyplot as plt

import parameters
import dynamics as dyn
import common
import plots
import tracking_LQR


def run(prm):
  """
    Tracking of the trajectory generated in Task 2 (main_task2.py) from perturbed initial conditions

    Saves x_lqr.npy, u_lqr.npy in prm.save_dir: the closed loop from the nominal perturbation

    Return
      - dictionary with 'runs': list of (label, xx, uu), one for each initial condition
  """
  xx_gen, uu_gen = common.load_generated_trajectory(prm)

  runs = []
  for scale in [0.5, 1.0, 2.0]:
    x0 = xx_gen[:,0] + scale * prm.tracking_x0_perturbation
    # TODO 20: compute xx, uu: LQR tracking of (xx_gen, uu_gen) from x0 (tracking_LQR.track)
    raise NotImplementedError("TODO 20: compute xx, uu: LQR tracking of (xx_gen, uu_gen) from x0 (tracking_LQR.track)")
    runs.append((f'perturbation x {scale}', xx, uu))

  common.save_array(prm, 'x_lqr.npy', runs[1][1])
  common.save_array(prm, 'u_lqr.npy', runs[1][2])

  figures = {}
  figures['task3_trajectory'] = plots.plot_trajectory(runs[1][1], runs[1][2], xx_gen, uu_gen, dyn.dt,
                                                    'LQR tracking', labels=('LQR', 'generated'))
  figures['task3_error'] = plots.plot_tracking_error(runs, xx_gen, dyn.dt, 'LQR: tracking error')

  common.finalize(prm, figures)

  return {'runs': runs}


if __name__ == '__main__':
  run(parameters)
