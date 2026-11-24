#
# OPTCON-RL 2026/2027
# Trajectory tracking via LQR
#

import numpy as np

from solver_LQR import ltv_LQR


def linearize_trajectory(dyn, xx_gen, uu_gen):
  """
    Linearization of the dynamics about the trajectory (xx_gen, uu_gen)

    Return
      - AA (ns, ns, TT), BB (ns, ni, TT)

    Example
      AA, BB = tracking_LQR.linearize_trajectory(dyn, xx_gen, uu_gen)
      A_0, B_0 = AA[:,:,0], BB[:,:,0]           # linearization at the first sample
  """
  ns, TT = xx_gen.shape
  ni = uu_gen.shape[0]

  AA = np.zeros((ns, ns, TT))
  BB = np.zeros((ns, ni, TT))

  for tt in range(TT-1):
    dfx, dfu = dyn.dynamics(xx_gen[:,tt], uu_gen[:,tt])[1:]
    AA[:,:,tt] = dfx.T
    BB[:,:,tt] = dfu.T

  return AA, BB


def track(dyn, xx_gen, uu_gen, x0, prm):
  """
    Tracking of the trajectory (xx_gen, uu_gen) with the LQR feedback computed
    on the linearization of the dynamics about the trajectory

      u_t = u_gen_t + K_t (x_t - x_gen_t)

    Args
      - dyn: dynamics module
      - xx_gen (ns, TT), uu_gen (ni, TT): trajectory to be tracked
      - x0 (ns,): initial condition of the system
      - prm: parameters (QQt_reg, RRt_reg, QQT_reg)

    Return
      - xx (ns, TT), uu (ni, TT): closed-loop trajectory of the nonlinear system

    Example
      x0 = xx_gen[:,0] + prm.tracking_x0_perturbation
      xx, uu = tracking_LQR.track(dyn, xx_gen, uu_gen, x0, prm)
      error = np.linalg.norm(xx - xx_gen, axis=0)
  """
  ns, TT = xx_gen.shape
  ni = uu_gen.shape[0]

  AA, BB = linearize_trajectory(dyn, xx_gen, uu_gen)

  QQ = np.repeat(prm.QQt_reg[:,:,None], TT, axis=2)
  RR = np.repeat(prm.RRt_reg[:,:,None], TT, axis=2)
  SS = np.zeros((ni, ns, TT))

  # LQ problem without affine terms: only the gains are needed

  # TODO 18: compute KK (ni, ns, TT): gains of the LQR tracking controller (ltv_LQR on the linearization AA, BB)
  raise NotImplementedError("TODO 18: compute KK (ni, ns, TT): gains of the LQR tracking controller (ltv_LQR on the linearization AA, BB)")

  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0

  for tt in range(TT-1):
    # TODO 19: compute uu[:,tt]: feedback law of the LQR, from uu_gen, KK and the error xx - xx_gen at time tt
    raise NotImplementedError("TODO 19: compute uu[:,tt]: feedback law of the LQR, from uu_gen, KK and the error xx - xx_gen at time tt")
    xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], jac=False)[0]

  uu[:,-1] = uu[:,-2]     # not used, for plots only

  return xx, uu
