import numpy as np
import pandas as pd
import json
import time
import re


# Function importing Dataset
def importdata():
    # train_data = pd.read_csv(
    #     'training_set_rel3.tsv', skipinitialspace=True, header=None)
    data = pd.read_table('training_set_rel3.tsv', header=None, sep = "\t+")
    # print(data.head())
    print(data.shape)
    return data.values


# Function to split the dataset
# def splitdataset(data):
#     # Seperating the target variable
#     x = data[:, i:j]
#     y = data[:, 0]
#     # print(np.shape(x), np.shape(y))
#     return x, y

def feature_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return mu, sigma

def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x

def preprocessing(training_data):
    #split data till essay column
    # data = splitdataset(training_data)
    essay = []
    for row in training_data:
        essay.append(row[2])
        # print(row[2])
    return essay

def create_essay_chunks(list_of_essays):
    essay_counter = 1;
    dict_of_essays_with_chunks = dict()
    list_of_essays.pop(0)
    for essay in list_of_essays:
        list_of_chunks = list()
        list_of_sentences = essay.split(". ")
        for sentence in list_of_sentences:
            clean_sentence = sentence.replace(",", "").replace("!", "").replace("-", "").replace(":", "").replace(".","")
            list_of_chunks.append(clean_sentence)
        dict_of_essays_with_chunks[essay_counter] = list_of_chunks
        essay_counter += 1
    return dict_of_essays_with_chunks

def write_chunked_essay_to_file(dict_of_essays_with_chunks):
    wf = open("dict_of_chunked_essays.txt", "w")
    wf.write(str(dict_of_essays_with_chunks))


if __name__ == '__main__':
    start_time = time.time()
    training_data = importdata()
    # print(training_data[1,0:])
    list_of_essays = preprocessing(training_data)
    dict_of_essays_with_chunks = create_essay_chunks(list_of_essays)
    write_chunked_essay_to_file(dict_of_essays_with_chunks)
    end_time = time.time()
    print("time",end_time - start_time)
