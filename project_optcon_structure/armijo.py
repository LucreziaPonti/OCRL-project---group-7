#
# OPTCON-RL 2026/2027
# Newton's method for optimal control
# Armijo stepsize selection rule on the cost
#

import numpy as np
import matplotlib.pyplot as plt


def select_stepsize(cost_fn, JJ, descent_arm, stepsize0, cc, beta, max_iters, plot=False):
  """
    Stepsize selection via the Armijo rule (backtracking) on the cost

    The cost along the descent direction is a scalar function of the stepsize,

      cost_fn(gamma) = J(u^k + gamma * deltau^k)

    where each evaluation requires a forward simulation of the dynamics.

      accept the stepsize gamma if   cost_fn(gamma) <= JJ + cc * gamma * descent_arm
      otherwise                      gamma = beta * gamma

    Args
      - cost_fn: function of the stepsize only; cost_fn(gamma) is the cost of
        the trajectory obtained with the stepsize gamma
      - JJ: cost at the current iteration, i.e., cost_fn(0)
      - descent_arm: Armijo descent direction at the current iteration, i.e., the
        slope of the tangent of cost_fn at zero,  grad J(u^k)' deltau^k
      - stepsize0: initial stepsize
      - cc, beta: Armijo parameters
      - max_iters: maximum number of backtracking steps
        (if 0, stepsize0 is returned without any test)
      - plot: if True, show the descent plot

    Return
      - stepsize

    Example
      # (xx, uu): current trajectory;  KK, sigma, descent_arm, JJ: from descent_direction
      def cost_fn(stepsize):
        xx_temp, uu_temp = algo_newton_optcon.closed_loop_update(dyn, xx, uu, KK, sigma, stepsize)
        return cost.trajectory_cost(xx_temp, uu_temp, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT)

      stepsize = armijo.select_stepsize(cost_fn, JJ, descent_arm, prm.armijo_stepsize0, prm.armijo_c,
                                        prm.armijo_beta, prm.armijo_max_iters, plot=False)
  """
  stepsize = stepsize0
  stepsizes = []    # tested stepsizes
  costs_armijo = []  # cost at the tested stepsizes

  for ii in range(max_iters):

    JJ_temp = cost_fn(stepsize)

    stepsizes.append(stepsize)
    costs_armijo.append(JJ_temp)

    if JJ_temp > JJ + cc * stepsize * descent_arm or not np.isfinite(JJ_temp):
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
    # TODO 15: draw the Armijo plot in a new figure: cost_fn along the direction (steps, values), the line JJ + gamma*descent_arm, the Armijo line with cc, and the tested stepsizes
    raise NotImplementedError("TODO 15: draw the Armijo plot in a new figure: cost_fn along the direction (steps, values), the line JJ + gamma*descent_arm, the Armijo line with cc, and the tested stepsizes")

  return stepsize
