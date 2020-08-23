import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from sklearn.model_selection import cross_validate


tokenized_file = filename

f = open(tokenized_file, "r")
tokenized_text = f.readlines()
tokenized_list = list(tokenized_text)
longest_sentence = max([len(subl) for subl in tokenized_list])

# To do: pad/truncate sentence sequences such that sequence length is identical
# This ensures LSTM time steps are the same


def vectorize_word(tokenized_text, size):
    vector_model = Word2Vec(tokenized_text, size=size,
                            window=5, min_count=1, workers=4)
    vector_model.save("word2vec.model")
    return vector_model
# All words in vector_model are now 100 x 1 vectors of numbers (no need to pad this)
# Vector_model contains vectors for every distinct word in training corpus


def vector_sentence(tokenized_sentence):
    vector_sentence = [vector_model.wv[word] for word in tokenized_sentence]
    return vector_sentence

# Apply pad to each line of words


def padding(vector_sentence):
    for l in tokenized_list:
        difference = longest_sentence(tokenized_list) - len(l)
        vector_sentence = vector_sentence(l)
        for i in range(difference):
            zero_vector = np.zeros(100, 1)
            vector_sentence.insert(0, zero_vector)
    return vector_sentence







# Divide training, test, and validation data (k-fold cross?)
X =

# Defining LSTM neural architecture
model = Sequential()
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Configure loss and other metrics
# Loss is calculated by cross entropy, adam optimizer uses stochastic gradient descent
