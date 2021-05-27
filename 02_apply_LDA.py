import pickle
import numpy as np
from gensim.models import LdaMulticore


def get_theta_matrix(ldaModel, corpus):
    """
    Return the document-topics matrix (denoted theta).
    Eacn line of this matrix contains the topic distribution of the 
    corresponding document (i.e. the probability that the document was 
    assigned to the topic)
    """
    n = len(corpus)
    K = ldaModel.num_topics
    theta = np.zeros(shape=[n, K])
    for i, d in enumerate(corpus):
        l = ldaModel.get_document_topics(d)
        for t in l:
            theta[i, t[0]] = t[1]
    return theta

PATH2CORPUS = "./20newsgroup_corpus_gensim_5classes_1000docs.pickle"
PATH2DICTIONARY = "./dictionary_20newsgroups_5classes_1000docs.pickle"

# Load the dataset
with open(PATH2CORPUS, 'rb') as f:
    d2b_dataset = pickle.load(f)
# Load the dictionary
with open(PATH2DICTIONARY, 'rb') as f:
    dictionary = pickle.load(f)

# Compute the LDA decomposition using different number of topics
# here 15, 10 and 5
lda15 = LdaMulticore(
    corpus=d2b_dataset, num_topics=15, id2word=dictionary,
    workers=4, eval_every=None, passes=10, batch=True,
)

lda10 = LdaMulticore(
    corpus=d2b_dataset, num_topics=10, id2word=dictionary,
    workers=4, eval_every=None, passes=10, batch=True,
)

lda5 = LdaMulticore(
    corpus=d2b_dataset, num_topics=5, id2word=dictionary,
    workers=4, eval_every=None, passes=10, batch=True,
)

# Save the LDA decompositions
filenames = ["./lda15", "./lda10", "./lda5"]
objects = [lda15, lda10, lda5]

for i in range(len(filenames)):
    with open(filenames[i], 'wb') as f:
        pickle.dump(objects[i], f)

# Save the gamma matrix
gamma15 = lda15.inference(d2b_dataset)
np.save("./gamma15.npy", gamma15[0])
gamma10 = lda10.inference(d2b_dataset)
np.save("./gamma10.npy", gamma10[0])
gamma5 = lda5.inference(d2b_dataset)
np.save("./gamma5.npy", gamma5[0])

# Save the theta matrix
theta15 = get_theta_matrix(lda15, d2b_dataset)
np.save("./theta15.npy", theta15)
theta10 = get_theta_matrix(lda10, d2b_dataset)
np.save("./theta10.npy", theta10)
theta5 = get_theta_matrix(lda5, d2b_dataset)
np.save("./theta5.npy", theta5)
