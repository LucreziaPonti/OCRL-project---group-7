#
# OPTCON-RL 2026/2027
# Dynamics of the system: pendulum
# Discrete-time nonlinear dynamics (Forward Euler) and its gradients
#
# This is the only file that defines the system. To work on another system,
# change the dimensions, the parameters and the function dynamics.
#

import numpy as np

ns = 2    # number of states
ni = 1    # number of inputs

dt = 1e-2 # discretization stepsize - Forward Euler

# Dynamics parameters

mm = 0.4
ll = 0.2
bb = 1
gg = 9.81


def dynamics(xx, uu, jac=True):
  """
    Discrete-time nonlinear dynamics of a pendulum

      x_{t+1} = f(x_t, u_t),   x = [theta, theta_dot],   u = torque

    Args
      - xx \\in \\R^ns state at time t
      - uu \\in \\R^ni input at time t
      - jac: if False, the gradients are not computed

    Return
      - next state xx_{t+1}                        shape (ns,)
      - gradient of f wrt x, at xx,uu   (dfx)      shape (ns, ns)
      - gradient of f wrt u, at xx,uu   (dfu)      shape (ni, ns)
        (dfx and dfu are None if jac is False)

    Note: dfx and dfu are gradients, i.e., the linearization is
      A = dfx.T,   B = dfu.T

    Example
      xxp, dfx, dfu = dyn.dynamics(xx[:,tt], uu[:,tt])
      At, Bt = dfx.T, dfu.T
      xxp = dyn.dynamics(xx[:,tt], uu[:,tt], jac=False)[0]
  """
  xx = np.asarray(xx, dtype=float)
  uu = np.asarray(uu, dtype=float)

  xxp = np.zeros((ns,))

  xxp[0] = xx[0] + dt * xx[1]
  xxp[1] = xx[1] + dt * (- (gg / ll) * np.sin(xx[0]) - (bb / (mm * ll)) * xx[1] + (1 / (mm * (ll ** 2))) * uu[0])

  if not jac:
    return xxp, None, None

  # Gradient

  dfx = np.zeros((ns, ns))
  dfu = np.zeros((ni, ns))

  #df1
  dfx[0,0] = 1
  dfx[1,0] = dt

  dfu[0,0] = 0

  #df2
  dfx[0,1] = - dt * gg / ll * np.cos(xx[0])
  dfx[1,1] = 1 + dt * (- bb / (mm * ll))

  dfu[0,1] = dt / (mm * (ll ** 2))

  return xxp, dfx, dfu
