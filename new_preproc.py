import numpy as np
import pandas as pd
from pymystem3 import Mystem
from nltk.corpus import stopwords
from humor_recognition.data import load_gold, load_train, load_test
import re


alpha_size = 33
c = 0


def preproc_texts(data):

    def process_text(text):
        tokens = text.split(' ')

        def word_2_vec(word):
            res = np.zeros(alpha_size)
            if len(word) == 0:
                return res
            char_codes = list(map(lambda x: ord(x) - 1072, word))
            unique, counts = np.unique(char_codes, return_counts=True)
            res[unique] = counts
            return res

        return (np.array(list(map(word_2_vec, tokens))) ** 3).sum(axis=0)

    return np.array(list(map(process_text, data)))


def preproc(train_lemmas, test_lemmas):
    train_data = pd.Series(load_train())
    y_train = train_data.values
    #X_train = preproc_texts(train_lemmas)
    test_data = pd.Series(load_test())
    y_test = test_data.values
    X_test = preproc_texts(test_lemmas)
    #print(X_train)
    #print(X_train.shape)
    print(X_test)
    return y_train, X_test, y_test


