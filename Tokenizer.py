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

def cleanup(clean_data):
    clean_data = re.sub("(?:@|#)[^\s]+", "", clean_data)                                 #remove twitter handles
    clean_data = re.sub(r"&.+?;", "", clean_data)
    clean_data = re.sub(r"^http?:\/\/.*[\r\n]*", "", clean_data, flags=re.MULTILINE)       #remove urls
    clean_data = re.sub(r"\d+", " ", clean_data)                                            #remove all numbers
    clean_data = re.sub("[^\w\d()':;\s]+", " ", clean_data)                                     #remove punctuation except ; : ( )
    clean_data = clean_data.lower()
    return clean_data

def tokenize(tokenized_file, data):
    clean_data = cleanup(data)
    tokentext = word_tokenize(clean_data)
    #tokentext_no_sw = [word for word in tokentext if not word in stopwords.words()]
    print (str(tokentext) + '\n')
    #with open(tokenized_file, 'a') as f:
        #f.write(tokentext + '\n')

def get_raw_data(filename, tokenized_file):
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v) 
    val = columns['Text']
    for i in range(140, 155):
        print (val[i])
        tokenize(tokenized_file, val[i])
    print("############################################################")

filename = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\training_data.csv'
tokenized_file = 'C:\\Users\\Johnn\\Documents\\GitHub\\IgnitionHacks2020\\tokenized_data.txt'
get_raw_data(filename, tokenized_file)