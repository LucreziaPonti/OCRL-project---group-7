#
# OPTCON-RL 2026/2027
# Helpers shared by the main files: saving, loading and showing the results
#

import os

import numpy as np
import matplotlib.pyplot as plt



def save_array(prm, name, array):
  os.makedirs(prm.save_dir, exist_ok=True)
  np.save(os.path.join(prm.save_dir, name), array)


def load_generated_trajectory(prm):
  """
    Trajectory generated in Task 2 (main_task2.py)
  """
  x_file = os.path.join(prm.save_dir, 'x_gen.npy')
  if not os.path.isfile(x_file):
    raise FileNotFoundError(f"{x_file} not found: run main_task2.py first")
  return np.load(x_file), np.load(os.path.join(prm.save_dir, 'u_gen.npy'))


def finalize(prm, figures):
  """
    Save the figures (dictionary name -> figure) in prm.save_dir, then show or close them
  """
  os.makedirs(prm.save_dir, exist_ok=True)
  for name, fig in figures.items():
    fig.savefig(os.path.join(prm.save_dir, name + '.pdf'))
  if prm.show_plots:
    plt.show()
  else:
    plt.close('all')
