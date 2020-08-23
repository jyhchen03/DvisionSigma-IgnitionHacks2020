import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from sklearn.model_selection import KFold


token_file = filename
f = open(tokenized_file, "r")
token_text = f.readlines()
token_list = list(token_text)
longest_sentence = max([len(subl) for subl in token_list])


def vectorize_word(t_list, size):
    vector_model = Word2Vec(t_list, size=size,
                            window=5, min_count=1, workers=4)
    vector_model.save("word2vec.model")
    return vector_model
# Vector_model contains 100 x 1 vectors for every distinct word in training corpus

def padding(vector_sentence):
    for l in token_list:
        difference = longest_sentence - len(l)
        vector_sentence = vector_sentence(l)
        for i in range(difference):
            zero_vector = np.zeros(100, 1)
            vector_sentence.insert(0, zero_vector)
    return vector_sentence

def vectorize_text(t_list, size):
    vector_text = []
    for l in t_list:
        vector_sentence = [vectorize_word(t_list, size=size).wv[word] for word in l]
        vector_sentence = padding(vector_sentence)
        vector_text.append(vector_sentence)
    return vector_text


training_data = vectorize_text(token_list, 100)  # List of all sentences in vectorized form


# Divide training and labels
X_train = []
X_test = []
y_train = []
y_test = []


# Cross validation
num_folds = 5
fold_accuracy = []
fold_loss = []

inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

kf = KFold(n_splits=num_folds, shuffle=True)
fold = 1
for train, test in kf.split(inputs, targets):
    # Defining LSTM neural architecture, and maybe other candidate models
    model = Sequential()
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])  # Configure loss and other metrics
    trained_model = model.fit(X, y, validation_split=0.2, epochs=4)

    # Determining how well the model generalizes
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    fold_accuracy.append(scores[1])
    fold_loss.append(scores[0])

