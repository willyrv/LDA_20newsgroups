import numpy as np
import os

path2partialresults = "./temp"
n_points = 600

dist_polyn = np.zeros((n_points, n_points))
max_var_polyn = np.zeros((n_points, n_points))
for i in range(n_points):
    for j in range(i, n_points):
        path = os.path.join(path2partialresults, "./{}_{}_polyn_dist.txt".format(i, j))
        with open(path, 'r') as f:
            value = float(f.read())
            dist_polyn[i, j] = value
        path = os.path.join(path2partialresults, "./{}_{}_max_var.txt".format(i, j))
        with open(path, 'r') as f:
            value = float(f.read())
            max_var_polyn[i, j] = value
dist_polyn = dist_polyn + dist_polyn.T
np.save("./dist_polyn.npy", dist_polyn)
max_var_polyn = max_var_polyn + max_var_polyn.T
np.save("./max_var_polyn.npy", max_var_polyn)
