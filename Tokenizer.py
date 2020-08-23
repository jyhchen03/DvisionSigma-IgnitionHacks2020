import csv
import os
import string
import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

columns = defaultdict(list)
re_list = [r"(?:@|#)[^\s]+",r"&.+?;",r"(?:http|www.)\S+",r"\d+",r"[^\w\d()':;\s]+"]

def cleanup(clean_data):
    for i in range(len(re_list)):
        clean_data = re.sub(re_list[i], " ", clean_data)
    return clean_data.lower()

def tokenize(tokenized_file, data):         #TODO: possible stopwords
    clean_data = cleanup(data)
    tokentext = str(word_tokenize(clean_data))
    #print (str(tokentext) + '\n')
    with open(tokenized_file, 'a') as f:
        f.write(tokentext + '\n')

def get_raw_data(filename, tokenized_file):
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v) 
    val = columns['Text']
    for i in range(len(val)):
        #print (val[i])
        tokenize(tokenized_file, val[i])

filename = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\training_data.csv'
tokenized_file = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\tokenized_data.txt'
get_raw_data(filename, tokenized_file)