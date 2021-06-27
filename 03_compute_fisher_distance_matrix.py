import multiprocessing as mp
import numpy as np
import os
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import polygamma
from sklearn.datasets import fetch_20newsgroups

import geomstats.backend as gs
from geomstats.geometry.dirichlet_distributions import DirichletDistributions

# flake8: noqa

PATH2PARTIALRESULTS = "./partial_0-1-2-15_35"
PATH2GAMMA = "./gamma10_0-1-2-15_35.npy"

def cost_fun(param, point_start, point_end, n_times):
    """Computes the length of the parameterized curve t -> (x_1(t),...,x_n(t))
    where x_i are polynomial functions of t, such that (x_1(0),..., x_n(0))
    is point_start and (x_1(1),..., x_n(1)) is point_end, with middle
    coefficients given by param. The parameterized curve is computed at
    n_times discrete times.

    Parameters
    ----------
    param : array-like, shape=(degree - 1, dim)
    point_start: array-like, shape=(dim,)
    point_end : array-like, shape=(dim,)
    """
    degree = len(param) + 1
    dim = point_start.shape[0]
    dirichlet = DirichletDistributions(dim)

    last_coef = point_end - point_start - np.sum(param, axis=0)
    coef = np.vstack((point_start, param, last_coef))

    t = np.linspace(0., 1., n_times)
    t_curve = [t**i for i in range(degree + 1)]
    t_curve = np.stack(t_curve)
    curve = np.einsum('ij,ik->kj', coef, t_curve)

    t_velocity = [i * t**(i-1) for i in range(1, degree + 1)]
    t_velocity = np.stack(t_velocity)
    velocity = np.einsum('ij,ik->kj', coef[1:], t_velocity)

    if curve.min() < 0:
        return np.inf, np.inf, curve, np.nan

    velocity_sqnorm = dirichlet.metric.squared_norm(vector=velocity, base_point=curve)
    length = np.sum(velocity_sqnorm ** (1/2)) / n_times
    energy = np.sum(velocity_sqnorm) / n_times
    return energy, length, curve, velocity, velocity_sqnorm


def cost_jacobian(param, point_start, point_end, n_times):
    """Computes the jacobian of the cost function.

    Parameters
    ----------
    param : array-like, shape=(degree - 1, dim)
    point_start: array-like, shape=(dim,)
    point_end : array-like, shape=(dim,)
    """
    degree = param.shape[0] + 1
    dim = point_start.shape[0]
    dirichlet = DirichletDistributions(dim)

    last_coef = point_end - point_start - np.sum(param, 0)
    coef = np.vstack((point_start, param, last_coef))

    t = np.linspace(0., 1., n_times)
    t_position = [t**i for i in range(degree + 1)]
    t_position = np.stack(t_position)
    position = np.einsum('ij,ik->kj', coef, t_position)

    t_velocity = [i * t**(i-1) for i in range(1, degree + 1)]
    t_velocity = np.stack(t_velocity)
    velocity = np.einsum('ij,ik->kj', coef[1:], t_velocity)

    fac1 = np.stack([k * t**(k-1) - degree * t**(degree-1) for k in range(1, degree)])
    fac2 = np.stack([t**k - t**degree for k in range(1, degree)])
    fac3 = (velocity * polygamma(1, position)).T \
         - np.sum(velocity, 1) * polygamma(1, np.sum(position, 1))
    fac4 = (velocity**2 * polygamma(2, position)).T \
         - np.sum(velocity, 1)**2 * polygamma(2, np.sum(position, 1))

    res = (2 * np.einsum('ij,kj->ik', fac1, fac3) \
        + np.einsum('ij,kj->ik', fac2, fac4)) / n_times

    return res.T.reshape(dim * (degree-1))


def approx_distance(ind_tuple, path2partialresults=PATH2PARTIALRESULTS):
  """Computes distance between 20NewsGroups documents.

  Estimates Fisher-Rao distance between Dirichlet parameters of documents
  of 20NewsGroups with indices given by ind_tuple. The distance is computed
  by approximating the geodesic by a polynomial curve and taking its length.
  """
  i, j = ind_tuple
  point_start = points[i]
  point_end = points[j]
  dim = point_start.shape[0]

  def f2minimize(x):
    param = x.reshape((dim, degree - 1)).T
    res = cost_fun(param, point_start, point_end, n_times)
    cost = res[0]
    return cost

  def jacobian(x):
    param = x.reshape((dim, degree - 1)).T
    res = cost_jacobian(param, point_start, point_end, n_times)
    return res

  x0 = np.tile(np.arange(degree - 1), dim)
  sol = minimize(f2minimize, x0, method=method, jac=jacobian)
  opt_param = (sol.x).reshape((dim, degree - 1)).T
  res = cost_fun(opt_param, point_start, point_end, n_times)
  polyn_dist = res[1]
  polyn_curve = res[2]
  polyn_velocity = res[3]
  polyn_vnorm = res[4]**(1/2)
  max_var = (np.max(polyn_vnorm) - np.min(polyn_vnorm)) / np.mean(polyn_vnorm)
  path = os.path.join(path2partialresults, "./{}_{}_polyn_dist.txt".format(i, j))
  with open(path, 'w') as f:
    f.write(str(polyn_dist))
  path = os.path.join(path2partialresults, "./{}_{}_max_var.txt".format(i, j))
  with open(path, 'w') as f:
    f.write(str(max_var))
  return 0

def check_computed_pairs(path2partialresults="./temp"):
  l = os.listdir(path2partialresults)
  d1 = {}
  computed_pairs = {}
  for v in l:
    if v.endswith("polyn_dist.txt") or v.endswith("max_var.txt"):
      i, j, ignore1, ignore1 = v.split("_")
      if (i, j) in d1:
        computed_pairs[(int(i), int(j))] = 1
      d1[(i, j)] = 1
  return computed_pairs

# Load all documents and their Dirichlet parameters
np.set_printoptions(suppress=True)
points = np.load(PATH2GAMMA) ### Load the matrix

# Set parameters for the computation of distances between documents
degree = 6
method = None
n_times = 500

if not os.path.exists(PATH2PARTIALRESULTS):
    os.makedirs(PATH2PARTIALRESULTS)
# Check if some values have already been computed
computed_pairs = check_computed_pairs(PATH2PARTIALRESULTS)
# Compute the matrix of geodesic distances between documents
n_points = points.shape[0]
ind_list = []
for i in range(n_points):
  for j in range(i+1, n_points):
    if not (i, j) in computed_pairs:
      ind_list.append((i, j))

nb_cpu = 30
pool = mp.Pool(nb_cpu)
results = pool.map(approx_distance, [ind_tuple for ind_tuple in ind_list])
