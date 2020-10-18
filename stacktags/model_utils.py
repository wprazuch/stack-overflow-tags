from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def preprocess_input(title, body):
    data = ' '.join([title, body])

    data = np.array(data)
    data = np.expand_dims(data, axis=-1)

    with open('tokenizer.pck', 'rb') as handle:
        tokenizer = pickle.load(handle)

    data_tokenized = tokenizer.texts_to_matrix(data)
    return data_tokenized
