import numpy as np
import pandas as pd
import time
import nltk


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


if __name__ == '__main__':
    start_time = time.time()
    training_data = importdata()
    # print(training_data[1,0:])
    preprocessing(training_data)
    end_time = time.time()
    print("time",end_time - start_time)



