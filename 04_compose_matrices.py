import numpy as np
import os
import h5py
from numpy.core.fromnumeric import size

path2partialresults = "./partial_0-1-2-15_35"
n_points = 140

# Create the files
with h5py.File("./distances_matrix.hdf5", "w") as f:
    dset = f.create_dataset('distances', (n_points, n_points), 
                            dtype=np.float, data=np.zeros([n_points, n_points]))
with h5py.File("./max_var_matrix.hdf5", "w") as f:
    dset = f.create_dataset('max_variations', (n_points, n_points),
                            dtype=np.float, data=np.zeros([n_points, n_points]))

# Write the values to the distances matrix and the max_variation matrix
distance_file = h5py.File("./distances_matrix.hdf5", 'r+')
distance_matrix = distance_file['distances']
max_var_file = h5py.File("./max_var_matrix.hdf5", 'r+')
max_var_matrix = max_var_file['max_variations']
for i in range(n_points):
    for j in range(i+1, n_points):
        path = os.path.join(path2partialresults, "{}_{}_polyn_dist.txt".format(i, j))
        with open(path, 'r') as f:
            value = float(f.read())
            distance_matrix[i, j] = value
            distance_matrix[j, i] = value
        path = os.path.join(path2partialresults, "{}_{}_max_var.txt".format(i, j))
        with open(path, 'r') as f:
            value = float(f.read())
            max_var_matrix[i, j] = value
            max_var_matrix[j, i] = value
distance_file.close()
max_var_file.close()

def test_matrix(path2partialresults, matrix_filename, 
                    sufix = "_polyn_dist.txt", nvalues=100):
    """
    Test if the matrix have been correctly created, i.e. verify that the content
    of the file {papath2partialresults}/{i}_{j}_polyn_dist.txt be equal to the 
    value in Matrix[i, j].
    The matrix is stored in hdf5 format.
    ntimes: Number values to test
    """
    f = h5py.File(matrix_filename, 'r')
    data = f[list(f.keys())[0]]
    n = data.shape[0]
    pairs = np.random.randint(0, n, size=[nvalues, 2])
    nb_errors = 0
    for (i, j) in pairs:
        row, column = np.sort([i, j])
        if row == column:
            column+=1
        fname = os.path.join(path2partialresults, "{}_{}{}".format(row, column, sufix))
        with open(fname, 'r') as f:
            value = float(f.read())
        if data[row, column] != value:
            print("Problem with {}, {}".format(row, column))
            nb_errors +=1
    f.close()
    print("{} errors detected".format(nb_errors))

