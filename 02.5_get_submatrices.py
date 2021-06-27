import numpy as np
import pandas as pd

PATH2DATASET = "./20newsgroup_corpus_text.csv"
PATH2THETA = "./theta10.npy"
PATH2GAMMA = "./gamma10.npy"
targets_to_keep = [0, 1, 2, 15]
nb_of_documents_class = 35
PATH2FILTEREDTHETA = "./theta10_0-1-2-15_35.npy"
PATH2FILTEREDGAMMA = "./gamma10_0-1-2-15_35.npy"

data = pd.read_csv(PATH2DATASET)
data_filter1 = data[data["target"].isin(targets_to_keep)]
data_filter2 = data_filter1.groupby("target").head(n=nb_of_documents_class)
Ix = data_filter2.index
theta = np.load(PATH2THETA)
new_theta = theta[Ix, :]
np.save(PATH2FILTEREDTHETA, new_theta)
gamma = np.load(PATH2GAMMA)
new_gamma = gamma[Ix, :]
np.save(PATH2FILTEREDGAMMA, new_gamma)

