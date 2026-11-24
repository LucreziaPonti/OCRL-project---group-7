#
# OPTCON-RL 2026/2027
# Cost functions
#

import numpy as np
import control as ctrl


def input_barrier(uu, bounds):
  """
    Relaxed logarithmic barrier for the box constraint u_min <= u <= u_max on the input

      b(u) = eps * sum_i [ phi(u_max_i - u_i) + phi(u_i - u_min_i) ]

      phi(d) = -log(d)                                       if d >= delta
      phi(d) = -log(delta) + 1/2 [ ((d - 2 delta)/delta)^2 - 1 ]   if d < delta

    The quadratic part continues the logarithm below the distance delta from a bound, with the
    same value, slope and curvature at delta: the barrier is finite everywhere (also outside the
    box), smooth and convex, so that any initial guess can be used and Armijo works as usual.

    Args
      - uu (ni,): input
      - bounds: (u_min (ni,), u_max (ni,), eps, delta), with eps > 0 the weight of the barrier
        and 0 < delta < (u_max - u_min)/2

    Return
      - bb: barrier at uu
      - dbu (ni,): gradient
      - ddbu (ni, ni): Hessian (diagonal, positive)

    Example
      bounds = (np.array([-5.0]), np.array([5.0]), 0.1, 1e-2)     # u_min, u_max, eps, delta
      bb, dbu, ddbu = cost.input_barrier(uu[:,tt], bounds)
  """
  u_min, u_max, eps, delta = bounds

  # TODO 24: compute bb, dbu (ni,), ddbu (ni,): value, gradient and diagonal of the Hessian of the relaxed barrier
  raise NotImplementedError("TODO 24: compute bb, dbu (ni,), ddbu (ni,): value, gradient and diagonal of the Hessian of the relaxed barrier")

  return bb, dbu, np.diag(ddbu)


def stagecost(xx, uu, xx_ref, uu_ref, QQt, RRt, bounds=None):
  """
    Stage-cost

    Quadratic cost function
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)
    plus, if bounds is given, the barrier input_barrier(u, bounds) for the input box constraint

    Args
      - xx \\in \\R^ns state at time t
      - uu \\in \\R^ni input at time t
      - xx_ref \\in \\R^ns state reference at time t
      - uu_ref \\in \\R^ni input reference at time t
      - QQt (ns, ns), RRt (ni, ni) weights
      - bounds: None (default), or (u_min, u_max, eps, delta) for the barrier (see input_barrier)

    Return
      - ll: cost at xx,uu
      - dlx (ns,): gradient of l wrt x, at xx,uu
      - dlu (ni,): gradient of l wrt u, at xx,uu
      - ddlx (ns, ns): Hessian of l wrt x
      - ddlu (ni, ni): Hessian of l wrt u
      - dlxu (ni, ns): mixed second derivative

    Example
      ll, dlx, dlu, ddlx, ddlu, dlxu = cost.stagecost(xx[:,tt], uu[:,tt], xx_ref[:,tt], uu_ref[:,tt],
                                                       prm.QQt, prm.RRt)
  """
  ns, ni = QQt.shape[0], RRt.shape[0]

  dx = xx - xx_ref
  du = uu - uu_ref

  ll = 0.5 * dx @ QQt @ dx + 0.5 * du @ RRt @ du

  dlx = QQt @ dx
  dlu = RRt @ du

  ddlx = QQt
  ddlu = RRt
  dlxu = np.zeros((ni, ns))

  if bounds is not None:
    bb, dbu, ddbu = input_barrier(uu, bounds)
    ll, dlu, ddlu = ll + bb, dlu + dbu, ddlu + ddbu

  return ll, dlx, dlu, ddlx, ddlu, dlxu


def termcost(xT, xT_ref, QQT):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xT \\in \\R^ns state at time T
      - xT_ref \\in \\R^ns state reference at time T
      - QQT (ns, ns) weight

    Return
      - lT: cost at xT
      - dlTx (ns,): gradient of l_T wrt x, at xT
      - ddlTx (ns, ns): Hessian of l_T

    Example
      lT, dlTx, ddlTx = cost.termcost(xx[:,-1], xx_ref[:,-1], QQT)
  """
  dx = xT - xT_ref

  lT = 0.5 * dx @ QQT @ dx
  dlTx = QQT @ dx
  ddlTx = QQT

  return lT, dlTx, ddlTx


def trajectory_cost(xx, uu, xx_ref, uu_ref, QQt, RRt, QQT, bounds=None):
  """
    Cost of a state-input curve

    Args
      - xx (ns, TT), uu (ni, TT): curve (the last column of uu is not used)
      - xx_ref (ns, TT), uu_ref (ni, TT): reference curve
      - bounds: None (default), or the bounds of the input barrier (see stagecost)

    Return
      - JJ: sum of the stage costs for t = 0,...,TT-2 plus the terminal cost

    Example
      JJ = cost.trajectory_cost(xx, uu, xx_ref, uu_ref, prm.QQt, prm.RRt, QQT)
  """
  TT = xx.shape[1]

  JJ = 0
  for tt in range(TT-1):
    JJ += stagecost(xx[:,tt], uu[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt, bounds)[0]
  JJ += termcost(xx[:,-1], xx_ref[:,-1], QQT)[0]

  return JJ


def terminal_weight(dyn, xx_eq, uu_eq, QQt, RRt):
  """
    Terminal weight QQT as the solution of the DARE for the linearization
    of the system about the equilibrium (xx_eq, uu_eq)

    If the DARE cannot be solved, QQt is returned and a warning is printed.

    Return
      - QQT (ns, ns)

    Example
      # linearization about the last point of the reference curve (an equilibrium)
      QQT = cost.terminal_weight(dyn, xx_ref[:,-1], uu_ref[:,-1], prm.QQt, prm.RRt)
  """
  dfx, dfu = dyn.dynamics(xx_eq, uu_eq)[1:]
  AA = dfx.T
  BB = dfu.T

  try:
    QQT = np.asarray(ctrl.dare(AA, BB, QQt, RRt)[0])
    if not np.all(np.isfinite(QQT)):
      raise ValueError("non-finite solution")
  except Exception as err:
    print(f"Warning: DARE not solved ({err}), terminal weight set to QQt")
    QQT = QQt.copy()

  return QQT
