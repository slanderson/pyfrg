import numpy as np
import subprocess

import pdb

def load_xpost(fname):
  """ Loads an xpost file into a numpy array

  Args: 
    fname (string): a string holding the path to the xpost file to be loaded

  Returns:
    ndarray: a 3D ndarray holding the xpost data, with shape (num_nodes, dim_of_vector,
      num_vectors)
    ndarray: a 1D mdarray holding the float-valued tags associated with each vector in the
      xpost file
  
  """
  with open(fname) as f:

    # get number of nodes, vector dimension, and number of vectors
    f.readline()  # ignore header
    num_nodes = int(f.readline())
    f.readline()  # skip to first data point
    first_data = f.readline().split()
    dim = len(first_data)
    num_lines = sum(1 for _ in f) + 4 # we've already seen 4 lines
    num_vecs = int( (num_lines - 2) / (num_nodes + 1) )
    f.seek(0)

    data = np.empty((num_nodes, dim, num_vecs))
    tags = np.empty(num_vecs)
    
    for _ in range(2): f.readline()  # skip header lines
    for vec in range(num_vecs):
      tags[vec] = float(f.readline())
      for node in range(num_nodes):
        data[node, :, vec] = [float(i) for i in f.readline().split()]
    
  return data, tags 


def load_node_positions_top(fname):
  """ Loads the node positions from a top file

  Args: 
    fname (string): a string holding the path to the top file whose node positions should
      be loaded

  Returns:
    ndarray: a 2D ndarray with shape (num_nodes, 3) holding the node positions
  
  """
  with open(fname) as f:

    f.readline()  # ignore header
    positions_list = []
    for line in f:
      if line.lstrip()[0].isalpha(): break
      positions_list += [[float(i) for i in line.split()[1:]]]
      
  return np.array(positions_list)
