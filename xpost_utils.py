import numpy as np
import subprocess

import pdb

def load_xpost(fname):
  """Loads an xpost file into a numpy array

  Args: 
    fname (string): a string holding the path to the xpost file to be loaded

  Returns:
    ndarray: a 3D ndarray holding the xpost data, with shape (num_nodes, dim_of_state,
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


def stack_solution_vectors(sol_vecs):
  """Turns a 3D array of solution vectors into a 2D array by flattening solution vectors

  Running load_xpost() on a file containing several solution vectors will give a 3D array,
  which must be indexed by both the node number and the dimension-index into the state
  vector.  Then the third dimension indexes into a chosen 2D solution vector.  But it is
  often desirable to stack each solution vector into an actual 1D vector, where each
  element represents a DOF (a specific state vector value at a specific node).  For
  instance, to run an SVD on a set of solution vectors, the vectors all need to be in a
  normal 2D matrix.

  Args:
    sol_vecs (ndarray): a 3D ndarray with shape (num_nodes, dim_of_state, num_vectors)

  Returns:
    ndarray: A 2D array with shape (num_nodes*dim_of_state, num_vectors)
  """
  num_nodes = sol_vecs.shape[0]
  dim = sol_vecs.shape[1]
  num_vecs = sol_vecs.shape[2]

  return sol_vecs.reshape((num_nodes*dim, num_vecs))
  

def unstack_solution_vectors(stacked_vecs, dim):
  """Unstacks a 2D matrix of solution vectors into a 3D array indexable by state-vector dimension

  Sets of solution vectors can be most intuitively accessed when the array is 3D, with
  separate indices for node number, state-vector dimension, and vector number.  But to do
  linear algebra on a matrix of solution vectors, they must be stored as a 'matrix' 2D
  array, indexed by DOF.  This function transforms from a matrix of solution vectors
  indexed by DOF, to a 3D array indexed by node number and state-vector dimension. 

  Args:
    stacked_vecs (ndarray): a 2D ndarray with shape (num_nodes*dim_of_state, num_vectors)
    dim (int): the dimension of the state vector at each node (e.g. 5 for inviscid 3D
      simulations)

  Returns:
    ndarray: A 3D array with shape (num_nodes, dim_of_state, num_vectors)
  """
  num_nodes = int( stacked_vecs.shape[0] / dim )
  num_vecs = stacked_vecs.shape[1]

  return stacked_vecs.reshape( (num_nodes, dim, num_vecs) )


def load_node_positions_top(fname):
  """Loads the node positions from a top file

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
