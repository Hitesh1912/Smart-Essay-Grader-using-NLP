import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import time
from sklearn import preprocessing



# Function importing Dataset
def importdata():
    # train_data = pd.read_csv(
    #     'training_set_rel3.tsv', skipinitialspace=True, header=None)
    data = pd.read_table('training_set_rel3.tsv', header=None, sep = "\t+")
    # print(data.head())
    print(data.shape)
    return data.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    x = data[1:, 0:3]
    y = data[1:, 3:np.shape(data)[1]]
    print(np.shape(x), np.shape(y))
    return x, y

def feature_normalization(x):
    mu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    return mu, sigma

def normalization(x,mu,sigma):
    x = np.subtract(x, mu)
    x = np.divide(x, sigma)
    return x

def preprocess_data(training_data):
    #split data till essay column
    # data = splitdataset(training_data)
    essay = []
    for row in training_data:
        essay.append(row[2])
        # print(row[2])
    return essay

def create_essay_chunks(list_of_essays):
    essay_counter = 1
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


def clean_text(text):
    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = nltk.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text








if __name__ == '__main__':
    start_time = time.time()
    training_data = importdata()
    X_train, y_train = splitdataset(training_data)

    #normalize all the scores in y_train
    normalizer = preprocessing.StandardScaler()
    normalized_score = normalizer.fit_transform(y_train)

    # print(training_data[1,0:])
    list_of_essays = preprocess_data(training_data)
    dict_of_essays_with_chunks = create_essay_chunks(list_of_essays)
    write_chunked_essay_to_file(dict_of_essays_with_chunks)

    end_time = time.time()
    print("time",end_time - start_time)
