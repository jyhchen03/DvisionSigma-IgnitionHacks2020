import pandas as pd
import numpy as np
import csv
import os
import string
import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from collections import defaultdict


columns = defaultdict(list)
re_list = [r"(?:#|@)[^\s]+",r"&.+?;",r"(?:http|www.)\S+",r"[^<3]\d+",r"[^\w\d()'<|@:;\s]+"]

filename = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\training_data.csv'
tokenized_file = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\tokenized_data.txt'
dataframe = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\dataframe.txt'
labels = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\labels.txt'

def cleanup(clean_data):
    clean_data = re.sub(r"&lt;", "<", clean_data)
    for i in range(len(re_list)):
        clean_data = re.sub(re_list[i], " ", clean_data)
    return clean_data.lower()

def tokenize(tokenized_file, data):         #TODO: possible stopwords
    clean_data = cleanup(data)
    tknzr = TweetTokenizer(reduce_len=True)
    tokentext = tknzr.tokenize(clean_data)
    #print (str(tokentext) + '\n')
    #with open(tokenized_file, 'a') as f:
    #    f.write(tokentext + '\n')
    return tokentext

def get_tokenized_data(tokenized_file, data):
    data_list=[]
    for i in range(len(data)):
        #print (data[i])
        data_list.append(tokenize(tokenized_file, data[i]))
    return data_list

def get_label_column(data):
    label_list=[]
    x = lambda a: label_list.append(int(a))
    for i in range(len(data)):
        x(data[i])
    return label_list

def write_labels():
    with open(labels,'a') as l:
        l.write(str(label))

with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k,v) in row.items():
            columns[k].append(v)

text = get_tokenized_data(tokenized_file, columns['Text'])
label = get_label_column(columns['Sentiment'])

#write_labels()
pd.set_option("display.max_rows", None)
dataframe_list = pd.DataFrame(np.column_stack([label,text]),columns=['Sentiment','Text'])
with open(dataframe,'a') as frame:
    frame.write(str(dataframe_list))

