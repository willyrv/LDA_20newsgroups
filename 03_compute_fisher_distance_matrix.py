import multiprocessing as mp
import numpy as np

from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import polygamma
from sklearn.datasets import fetch_20newsgroups

import geomstats.backend as gs
from geomstats.geometry.dirichlet_distributions import DirichletDistributions

# flake8: noqa


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


def approx_distance(ind_tuple):
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

  return polyn_dist, max_var


# Load all documents and their Dirichlet parameters
np.set_printoptions(suppress=True)
data = loadmat('examples/gamma_10.mat') ###### A REMPLACER
all_points = data['gamma10']
newsgroups = fetch_20newsgroups()

# Concentrate on a few classes
n_classes = 20
n_per_class = 100
indices = []
labels = []
for i in range(n_classes):
    indices_class_i = np.nonzero(newsgroups['target']==i)[0]
    indices.append(indices_class_i[:n_per_class])
    labels.append(i * np.ones(n_per_class))
indices = np.hstack(indices)
labels = np.hstack(labels)
labels = labels[np.argsort(indices)]
indices = np.sort(indices)
points = all_points[indices]

# Set parameters for the computation of distances between documents
degree = 6
method = None
n_times = 500

# Compute the matrix of geodesic distances between documents
n_points = n_classes * n_per_class
ind_list = []
for i in range(n_points):
    ind_list += [(i, j) for j in range(i+1, n_points)]

pool = mp.Pool(mp.cpu_count())
results = pool.map(approx_distance, [ind_tuple for ind_tuple in ind_list])

dist_polyn = np.zeros((n_points, n_points))
max_var_polyn = np.zeros((n_points, n_points))
counter = 0
for ind_tuple in ind_list:
  i, j = ind_tuple
  dist_polyn[i, j] = results[counter][0]
  max_var_polyn[i, j] = results[counter][1]
  counter += 1
