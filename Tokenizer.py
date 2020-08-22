import csv
import os
import re
from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import word_tokenize
from collections import defaultdict

columns = defaultdict(list)

def get_raw_data(filename, tokenized_file):
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v)
    val = columns['Text']
    for i in range(len(val)):
        tokenize(tokenized_file, val[i])

def tokenize(tokenized_file, data):
    with open(tokenized_file, 'a') as f:
        f.write(data + '\n')

filename = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\training_data.csv'
tokenized_file = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\tokenized_data.txt'
get_raw_data(filename, tokenized_file)