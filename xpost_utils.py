import numpy as np
import os

import pdb


class NotImplementedError(Exception):
    """
    Exception raised when a non-implemented feature is requested

    Attributes:
        expression: The input that causes the error
        message: Explanation of error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message



def load_xpost(fname):
  """Loads an xpost file into a numpy array

  Args: 
    fname (string): a string holding the path to the xpost file to be loaded

  Returns:
    ndarray: a 3D ndarray holding the xpost data, with shape (num_nodes, dim_of_state,
      num_vectors)
    ndarray: a 1D ndarray holding the numeric tags associated with each vector in the
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
    tag_list = []
    
    for _ in range(2): f.readline()  # skip header lines
    for vec in range(num_vecs):

      tag_str = f.readline() 
      tag_num = None
      try:
        tag_num = int(tag_str)
      except ValueError:
        tag_num = float(tag_str)
      tag_list += [tag_num]

      for node in range(num_nodes):
        data[node, :, vec] = [float(i) for i in f.readline().split()]
    
  tags = np.array(tag_list)
  return data, tags 


def write_xpost(fname, vecs, tags):
    """
    Writes an xpost file given a set of vectors and tags

    TODO: make compatible with fully general xpost headers
    TODO: document
    """
    num_nodes = vecs.shape[0]
    if len(vecs.shape) == 1:
        vecs = np.expand_dims(vecs, 1)
    if len(vecs.shape) == 2:
        vecs = np.expand_dims(vecs, 1)
    num_dims = vecs.shape[1]
    num_vecs = vecs.shape[2]

    var_type = 'Scalar' if num_dims == 1 else 'Vector'

    base=os.path.basename(fname)
    vec_name = os.path.splitext(base)[0]
    xpost_header =  '{} {} under load for FluidNodes\n'.format(var_type, vec_name)
    
    with open(fname, 'w') as f:
        f.write(xpost_header)
        f.write('{}\n'.format(num_nodes))

        for vec_num in range(num_vecs):

            f.write('{}\n'.format(tags[vec_num]))
            for node_num in range(num_nodes):
                node_arr = vecs[node_num, :, vec_num]
                dof_strings = ['{: .4E}'.format(x) for x in node_arr]
                node_string = ' '.join(dof_strings)
                f.write('{}\n'.format(node_string))


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
  

def unstack_solution_vectors(stacked_vecs, dim=5):
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


def conservative_to_primitive(cons_mat, 
                              desired_primitive="pressure", 
                              dimensionalize=False,
                              gamma=1.4,
                              p_inf=100000, 
                              rho_ref=1.3,
                              M_ref=0.4):
    """
    Converts an array of solution vectors into an array of scalars corresponding to a
    primitive variable, e.g. pressure/density/Mach

    TODO: document
    """
    # define primitive variables and intermediate variables
    def ref_velocity(M_ref, gamma, p_inf, rho_ref):
        return M_ref * np.sqrt(gamma * p_inf / rho_ref)
    def density(cons_mat, dimensionalize=False):
        return cons_mat[:, 0, :] * (rho_ref if dimensionalize else 1)
    def velocity(cons_mat, dimensionalize=False):
        v_ref = ref_velocity(M_ref, gamma, p_inf, rho_ref)
        v = cons_mat[:, 1:4, :] / np.expand_dims(cons_mat[:, 0, :], 1) * (v_ref if dimensionalize else 1)
        return v
    def velocity_mag(cons_mat, dimensionalize=False):
        v = velocity(cons_mat, dimensionalize=dimensionalize)
        return np.sqrt(np.linalg.norm(v, ord=2, axis=1))
    def total_energy(cons_mat):
        return cons_mat[:, 4, :]
    def pressure(cons_mat, dimensionalize=False):
        v_ref = ref_velocity(M_ref, gamma, p_inf, rho_ref)
        rho = cons_mat[:, 0, :]
        v = cons_mat[:, 1:4, :] / np.expand_dims(rho, 1)
        v_squared = np.square(v).sum(axis=1)
        E = cons_mat[:, 4, :]
        p = (gamma - 1) * (E - 0.5 * rho * v_squared) * (rho_ref * v_ref**2 if dimensionalize else 1) 
        return p
    def sound_speed(cons_mat, dimensionalize=False):
        p = pressure(cons_mat, dimensionalize=dimensionalize)
        rho = density(cons_mat, dimensionalize=dimensionalize)
        c = np.sqrt(gamma * p / rho)  
        return c
    def mach(cons_mat):
        p = pressure(cons_mat, dimensionalize=False)
        rho = density(cons_mat, dimensionalize=False)
        v_mag = velocity_mag(cons_mat, dimensionalize=False)
        M = np.sqrt( (rho * v_mag**2 ) / (gamma * p))
        return M

    if desired_primitive == 'density':
        return density(cons_mat, dimensionalize=dimensionalize)
    elif desired_primitive == 'velocity':
        return velocity(cons_mat, dimensionalize=dimensionalize)
    elif desired_primitive == 'pressure':
        return pressure(cons_mat, dimensionalize=dimensionalize)
    elif desired_primitive == 'sound_speed':
        return sound_speed(cons_mat, dimensionalize=dimensionalize)
    elif desired_primitive == 'Mach':
        return mach(cons_mat)
    else:
        raise NotImplementedError(desired_primitive,
                                  'Conversion to this primitive currently not supported')

