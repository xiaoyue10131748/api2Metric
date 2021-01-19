import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from text_classifier.code.w2v import get_w2v
from gensim.models import FastText

names = ["class", "title", "content"]


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio=1, n_class=15, names=names, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    api = pd.Series(shuffle_csv["title"])
    y = pd.Series(shuffle_csv["class"])

    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y,api


def data_preprocessing(train, test, max_len):
    """transform to one-hot idx vector by VocabularyProcessor"""
    """VocabularyProcessor is deprecated, use v2 instead"""
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size





def data_preprocessing_v2(train, test, max_len, max_words=50000):
    embeddings_dictionary = get_w2v()
    embedding_matrix = np.zeros((50002, 300))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)

    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')

    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        else:
            word=word[:-1]

            while len(word) >0 and embedding_vector is None:
                embedding_vector =  embeddings_dictionary.get(word)
                word=word[:-1]

            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                print("not in the word embedding :: " +word)


    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2,embedding_matrix

def sentence_to_vector(sen,embeddings_dictionary,max_len):
    i = 0
    sen_pad=np.zeros((max_len,300),dtype="float32")
    sen_list=str(sen).split()
    while i < max_len:
        if i < len(sen_list):
            word = sen_list[i]
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                sen_pad[i] = embedding_vector
            else:
                word = word[:-1]

                while len(word) > 0 and embedding_vector is None:
                    embedding_vector = embeddings_dictionary.get(word)
                    word = word[:-1]

                if embedding_vector is not None:
                    sen_pad[i] = embedding_vector
                else:
                    print("not in the word embedding :: " + word)
        else:
            sen_pad[i] = np.zeros(300)
        i+=1
    return sen_pad


def data_preprocessing_v3(train, test, max_len, max_words=50000):
    embeddings_dictionary = get_w2v()
    train_size=len(train)
    train_pad=np.zeros((train_size,max_len,300),dtype="float32")
    for index,sen in enumerate(train):
        train_pad[index] = sentence_to_vector(sen, embeddings_dictionary, max_len)

    test_size=len(test)
    test_pad=np.zeros((test_size,max_len,300))
    for index,sen in enumerate(test):
        test_pad[index] = sentence_to_vector(sen, embeddings_dictionary, max_len)

    return train_pad,test_pad



def data_preprocessing_v4(test, max_len, max_words=50000):
    embeddings_dictionary = get_w2v()
    test_size=len(test)
    test_pad=np.zeros((test_size,max_len,300))
    for index,sen in enumerate(test):
        test_pad[index] = sentence_to_vector(sen, embeddings_dictionary, max_len)
    return test_pad


def keyword_preprocessing_v3(keyword_list,max_len):
    embeddings_dictionary = get_w2v()
    keyword_embeding = sentence_to_vector(keyword_list[0], embeddings_dictionary, max_len)
    return keyword_embeding


'''
def data_preprocessing_v2(train, test, max_len, max_words=50000):

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    # split text into a list of words.
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2
'''


def keyword_preprocessing_v2(keyword_list):
    embeddings_dictionary = get_w2v()
    embedding_matrix = np.zeros((50002, 300))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100)
    tokenizer.fit_on_texts(keyword_list[0])

    keyword_idx = tokenizer.texts_to_sequences(keyword_list)

    keyword_padded = pad_sequences(keyword_idx, maxlen=100, padding='post', truncating='post')

    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return keyword_padded, embedding_matrix


def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size, ):
    while True:
        shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
        for x in fill_feed_dict_once(shuffled_X, shuffled_Y, batch_size):
            yield x

def fill_feed_dict_once(data_X, data_Y, batch_size, ):
    """Generator to yield batches"""
    # Shuffle data first.
    #         # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        y_batch = np.int32(y_batch)
        weight = np.int32(y_batch == 1)
        # print(y_batch)
        yield np.int32(x_batch), y_batch, np.ones(len(y_batch)) + weight * 3.0
