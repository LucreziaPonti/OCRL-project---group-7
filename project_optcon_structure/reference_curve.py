 #
# OPTCON-RL 2026/2027
# Reference (desired) curves for the pendulum: step (Task 1) and smooth (Task 2)
#
# This file depends on the system. Both curves start and finish at equilibria; the step
# moves between two equilibria, the smooth curve can be any smooth state-input curve
# with that property.
#

import numpy as np

import dynamics as dyn


def step_reference(TT, prm):
  """
    Desired state-input curve between two equilibria of the pendulum, with a step at TT/2

    Args
      - TT: number of samples
      - prm: parameters (reference_angle_initial_deg, reference_angle_final_deg)

    Return
      - xx_ref (ns, TT), uu_ref (ni, TT)

    Example
      TT = int(round(prm.tf / dyn.dt))
      xx_ref, uu_ref = reference_curve.step_reference(TT, prm)
  """
  theta_0 = np.deg2rad(prm.reference_angle_initial_deg)
  theta_T = np.deg2rad(prm.reference_angle_final_deg)

  ss = (np.arange(TT) >= TT//2).astype(float)            # profile from 0 to 1: a step at TT/2

  xx_ref = np.zeros((dyn.ns, TT))
  uu_ref = np.zeros((dyn.ni, TT))

  xx_ref[0,:] = theta_0 + (theta_T - theta_0) * ss       # angle; the velocity is zero

  # Input that keeps the pendulum at equilibrium at the angle of the curve
  uu_ref[0,:] = dyn.mm * dyn.gg * dyn.ll * np.sin(xx_ref[0,:])

  return xx_ref, uu_ref


def smooth_reference(TT, prm):
  """
    Smooth desired state-input curve: any smooth curve that starts and finishes at an
    equilibrium of the system (e.g., a shape drawn in the workspace, a circular path, ...);
    it does not need to be a trajectory of the system, nor to connect the equilibria of
    step_reference.

    For the pendulum, as an example: a smooth transition of the angle, with the fifth-order
    polynomial 10 s^3 - 15 s^4 + 6 s^5 (zero velocity and acceleration at both ends) on the
    fraction transition_fraction of the horizon, centered at TT/2, and the input that keeps
    the pendulum at equilibrium at each angle

    Args
      - TT: number of samples
      - prm: parameters (reference_angle_initial_deg, reference_angle_final_deg, transition_fraction)

    Return
      - xx_ref (ns, TT), uu_ref (ni, TT)

    Example
      TT = int(round(prm.tf / dyn.dt))
      xx_ref, uu_ref = reference_curve.smooth_reference(TT, prm)
  """
  # TODO 17: compute xx_ref (ns, TT), uu_ref (ni, TT): smooth reference curve, starting and finishing at equilibria
  raise NotImplementedError("TODO 17: compute xx_ref (ns, TT), uu_ref (ni, TT): smooth reference curve, starting and finishing at equilibria")

  return xx_ref, uu_ref
