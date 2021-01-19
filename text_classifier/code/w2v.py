from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os
import pickle


def get_w2v():

    wv = KeyedVectors.load("../../../w2v/word2vec.wordvectors", mmap='r')
    if not os.path.exists("../../../w2v/embeddings_dictionary"):
        embeddings_dictionary = {}
        for v in wv.wv.vocab:
            embeddings_dictionary[v] = wv[v]

        f = open('../../../w2v/embeddings_dictionary', 'wb')
        pickle.dump(embeddings_dictionary, f)
        f.close()
        return embeddings_dictionary

    else:
        f = open('../../../w2v/embeddings_dictionary', 'rb')
        data = pickle.load(f)
        print(len(data))
        return data


