#
# OPTCON-RL 2026/2027
# SQP for optimal control
# Armijo stepsize selection rule on the merit function
#

import numpy as np
import matplotlib.pyplot as plt


def select_stepsize(merit_fn, M0, DM0, stepsize0, cc, beta, max_iters, plot=False):
  """
    Stepsize selection via the Armijo rule (backtracking) on the merit function

    The SQP iterates are not trajectories, so the decrease is measured on the
    merit function  M1(z) = J(z) + s * ||h(z)||_1  along the direction d^k,

      merit_fn(gamma) = M1(z^k + gamma * d^k)

      accept the stepsize gamma if   merit_fn(gamma) <= M0 + cc * gamma * DM0
      otherwise                      gamma = beta * gamma

    Args
      - merit_fn: function of the stepsize only; merit_fn(gamma) is the merit
        function at the point reached with the stepsize gamma
      - M0: merit function at the current iterate, i.e., merit_fn(0)
      - DM0: directional derivative of the merit function along d^k,
        grad J(z^k)' d^k - s * ||h(z^k)||_1
      - stepsize0: initial stepsize
      - cc, beta: Armijo parameters
      - max_iters: maximum number of backtracking steps
        (if 0, stepsize0 is returned without any test)
      - plot: if True, show the descent plot

    Return
      - stepsize

    Example
      # (xx, uu): current iterate;  (dx, du): SQP direction;  ss: penalty of the merit function
      def merit_fn(stepsize):
        xx_temp = xx + stepsize * dx
        uu_temp = uu + stepsize * du
        JJ_temp = cost.trajectory_cost(xx_temp, uu_temp, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT)
        return JJ_temp + ss * np.sum(np.abs(algo_sqp_optcon.dynamics_defect(dyn, xx_temp, uu_temp)))

      stepsize = armijo_merit.select_stepsize(merit_fn, M0, DM0, prm.armijo_stepsize0, prm.armijo_c,
                                              prm.armijo_beta, prm.armijo_max_iters, plot=False)
  """
  stepsize = stepsize0
  stepsizes = []    # tested stepsizes
  merits_armijo = []  # merit function at the tested stepsizes

  for ii in range(max_iters):

    M_temp = merit_fn(stepsize)

    stepsizes.append(stepsize)
    merits_armijo.append(M_temp)

    if M_temp > M0 + cc * stepsize * DM0 or not np.isfinite(M_temp):
      # update the stepsize
      stepsize = beta * stepsize
    else:
      break
  else:
    if max_iters > 0:
      print(f'WARNING: no stepsize was found with armijo rule! Using stepsize = {stepsize:.2e}')

  ############################
  # Descent plot
  ############################

  if plot:
    # TODO 16: draw the Armijo plot in a new figure: merit_fn along the direction (steps, values), the line M0 + gamma*DM0, the Armijo line with cc, and the tested stepsizes
    raise NotImplementedError("TODO 16: draw the Armijo plot in a new figure: merit_fn along the direction (steps, values), the line M0 + gamma*DM0, the Armijo line with cc, and the tested stepsizes")

  return stepsize
