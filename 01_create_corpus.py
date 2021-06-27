from string import punctuation
import nltk
from nltk import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

PATH2CORPUS = "./20newsgroup_corpus_gensim.pickle"
PATH2DICTIONARY = "./dictionary_20newsgroups_gensim.pickle"
PATH2CORPUSTEXT = "./20newsgroup_corpus_text.csv"

newsgroups = fetch_20newsgroups()
nltk.download('stopwords')
targets_to_keep = [1, 2, 7, 8, 13, 14]
nb_of_documents_class = 100

data = pd.DataFrame({"text":newsgroups.data, "target": newsgroups.target})
eng_stopwords = set(stopwords.words('english'))

tokenizer = RegexpTokenizer(r'\s+', gaps=True)
stemmer = PorterStemmer()
translate_tab = {ord(p): u" " for p in punctuation}

def text2tokens(raw_text):
    """Split the raw_text string into a list of stemmed tokens."""
    clean_text = raw_text.lower().translate(translate_tab)
    tokens = [token.strip() for token in tokenizer.tokenize(clean_text)]
    tokens = [token for token in tokens if token not in eng_stopwords]
    # stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # return [token for token in stemmed_tokens if len(token) > 2]  # skip short tokens
    return [token for token in tokens if len(token) > 2]  # skip short tokens

dataset = [text2tokens(txt) for txt in list(data['text'].values)]  # convert documents to lists of tokens

dictionary = Dictionary(documents=dataset, prune_at=None)
dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None)  # use Dictionary to remove un-relevant tokens
dictionary.compactify()

d2b_dataset = [dictionary.doc2bow(doc) for doc in dataset]  # convert list of tokens to bag of word representation

import pickle
# Save the corpus representation in gensim format
# and the corresponding dictionary
with open(PATH2CORPUS, 'wb') as f:
    pickle.dump(d2b_dataset, f)

with open(PATH2DICTIONARY, 'wb') as f:
    pickle.dump(dictionary, f)

# Save the text documents
data.to_csv(PATH2CORPUSTEXT)

