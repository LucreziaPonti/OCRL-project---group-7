#
# OPTCON-RL 2026/2027
# Affine Linear Quadratic (LQ) optimal control solver
# Linear time-varying dynamics, finite horizon
#

import numpy as np


def ltv_LQR(AA, BB, QQ, RR, SS, QQT, qq, rr, qqT, cc, x0):
  """
    Solution of the affine LQ problem

      min  sum_{t=0}^{TT-2} [ q_t' dx_t + r_t' du_t
                              + 1/2 dx_t' Q_t dx_t + 1/2 du_t' R_t du_t + du_t' S_t dx_t ]
           + q_T' dx_T + 1/2 dx_T' Q_T dx_T

      subj.to  dx_{t+1} = A_t dx_t + B_t du_t + c_t,   t = 0,...,TT-2
               dx_0 = x0

    Args
      - AA (ns, ns, TT), BB (ns, ni, TT): dynamics matrices
      - QQ (ns, ns, TT), RR (ni, ni, TT), SS (ni, ns, TT): quadratic cost
      - QQT (ns, ns): terminal quadratic cost
      - qq (ns, TT), rr (ni, TT), qqT (ns,): affine cost
      - cc (ns, TT): affine term of the dynamics
      - x0 (ns,): initial condition

    The last element along the time axis of the stage quantities is not used.

    Return
      - KK (ni, ns, TT): feedback gains
      - sigma (ni, TT): feedforward terms,   du_t = KK_t dx_t + sigma_t
      - dx (ns, TT), du (ni, TT): optimal state-input curve
      - lmbd (ns, TT): costates (multipliers of the dynamics constraints),
          lmbd_t = P_t dx_t + p_t

    Example
      # descent direction of Newton's method at the trajectory (xx, uu)
      AA, BB, QQ, RR, SS, qq, rr, qqT, JJ = algo_newton_optcon.linearize(dyn, xx, uu, xx_ref, uu_ref,
                                                                    prm.QQt, prm.RRt, QQT)
      cc = np.zeros((dyn.ns, TT))       # (xx, uu) is a trajectory: no affine term in the dynamics
      x0 = np.zeros((dyn.ns,))          # the initial state does not change
      KK, sigma, dx, du, lmbd = solver_LQR.ltv_LQR(AA, BB, QQ, RR, SS, QQT, qq, rr, qqT, cc, x0)
  """
  ns, ni, TT = BB.shape

  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))

  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))

  dx = np.zeros((ns, TT))
  du = np.zeros((ni, TT))
  lmbd = np.zeros((ns, TT))

  PP[:,:,-1] = QQT
  pp[:,-1] = qqT

  # Solve the (augmented) Riccati equation backward in time

  for tt in reversed(range(TT-1)):

    At = AA[:,:,tt]
    Bt = BB[:,:,tt]
    Qt = QQ[:,:,tt]
    Rt = RR[:,:,tt]
    St = SS[:,:,tt]
    qt = qq[:,tt]
    rt = rr[:,tt]
    ct = cc[:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1] + PPtp @ ct

    # TODO 1: compute KK[:,:,tt], sigma[:,tt], PP[:,:,tt], pp[:,tt]: backward step of the augmented Riccati equation at time tt
    raise NotImplementedError("TODO 1: compute KK[:,:,tt], sigma[:,tt], PP[:,:,tt], pp[:,tt]: backward step of the augmented Riccati equation at time tt")

  # Forward simulation of the optimal closed-loop system

  dx[:,0] = x0

  for tt in range(TT-1):
    du[:,tt] = KK[:,:,tt] @ dx[:,tt] + sigma[:,tt]
    dx[:,tt+1] = AA[:,:,tt] @ dx[:,tt] + BB[:,:,tt] @ du[:,tt] + cc[:,tt]

  for tt in range(TT):
    lmbd[:,tt] = PP[:,:,tt] @ dx[:,tt] + pp[:,tt]

  return KK, sigma, dx, du, lmbd


