#
# OPTCON-RL 2026/2027
# Task 2 - Trajectory generation (II): smooth reference, Newton's method or SQP
#

import numpy as np
import matplotlib.pyplot as plt

import parameters
import dynamics as dyn
import common
import reference_curve
from main_task1 import optimal_transition


def run(prm):
  """
    Optimal transition between two equilibria along the smooth reference curve
    (reference_curve.smooth_reference), with the method chosen by prm.solver

    Saves x_gen.npy, u_gen.npy in prm.save_dir: the trajectory tracked in Task 3

    Return
      - dictionary with 'xx', 'uu', 'log', 'xx_ref', 'uu_ref'
  """
  TT = int(round(prm.tf / dyn.dt))
  xx_ref, uu_ref = reference_curve.smooth_reference(TT, prm)

  xx, uu, log = optimal_transition(prm, xx_ref, uu_ref, 'task2')

  common.save_array(prm, 'x_gen.npy', xx)
  common.save_array(prm, 'u_gen.npy', uu)

  return {'xx': xx, 'uu': uu, 'log': log, 'xx_ref': xx_ref, 'uu_ref': uu_ref}


if __name__ == '__main__':
  run(parameters)
