import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import time
from spellchecker import SpellChecker
from sklearn import preprocessing

spell = SpellChecker()

# Function importing Dataset
def importdata():
    # train_data = pd.read_csv(
    #     'training_set_rel3.tsv', skipinitialspace=True, header=None)
    data = pd.read_table('test_set.tsv', header=None, sep = "\t+", encoding='iso-8859-1', engine = 'python')
    # print(data.head())
    # print(data.shape)
    return data.values


# Function to split the dataset
def splitdataset(data):
    x = data[1:, 0:3]
    y = data[1:, 3:np.shape(data)[1]]
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
    dict_of_scores = dict()
    #split data till essay column
    # data = splitdataset(training_data)
    essay = []
    for row in training_data:
        essay.append(row[2])
        # print(row[2])
    return essay


def create_essay_chunks(list_of_essays):
    essay_counter = 1
    dict_of_essays = dict()
    dict_of_essays_with_chunks = dict()
    list_of_essays.pop(0)
    for essay in list_of_essays:
        dict_of_essays[essay_counter] = essay
        list_of_chunks = list()
        list_of_sentences = essay.split(". ")
        for sentence in list_of_sentences:
            # clean_sentence = sentence.replace(",", "").replace("!", "").replace("-", "").replace(":", "").replace(".","")
            # spelling_corrected_sentence = correct_spelling(sentence)
            clean_sentence = clean_text(sentence)
            list_of_chunks.append(clean_sentence)
        dict_of_essays_with_chunks[essay_counter] = list_of_chunks
        essay_counter += 1
    return dict_of_essays_with_chunks, dict_of_essays


def create_dict_of_essays(x_train):
    counter = 1
    dict_of_essays = dict()
    for row in x_train:
        dict_of_essays[counter] = row[2]
        counter += 1
    return dict_of_essays


def write_chunked_essay_to_file(dict_of_essays_with_chunks):
    wf = open("dict_of_chunked_essays_test.txt", "w", encoding='iso-8859-1')
    wf.write(str(dict_of_essays_with_chunks))


def write_essaydict_to_file(dict_of_essays):
    wf = open("dict_of_essays_test.txt", "w", encoding='iso-8859-1')
    wf.write(str(dict_of_essays))


def clean_text(text):
    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[0-9]", "", text)
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
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace('"', "")

    return text


def correct_spelling(sentence):
    words = sentence.split(" ")
    misspelled = spell.unknown(words)
    known_words = spell.known(words)
    unknown_words = spell.unknown(words)
    final_list = list()
    for word in words:
        if word in known_words:
            final_list.append(word)
        if word in unknown_words:
            final_list.append(spell.correction(word))
    sentence = ' '.join(final_list)
    print(sentence)
    return sentence


if __name__ == '__main__':
    start_time = time.time()
    dict_of_scores = dict()
    dict_of_essays = dict()
    training_data = importdata()
    x_train, y_train = splitdataset(training_data)
    x_train = np.array(x_train)

    essays = x_train[:, 2]
    counter = 1
    processed_essays = []

    for essay in essays:
        processed_essays.append(clean_text(essay))

    for essay in processed_essays:
        dict_of_essays[counter] = essay
        counter = counter + 1

    list_of_essays = preprocess_data(training_data)
    dict_of_essays_with_chunks, dict_of_essays = create_essay_chunks(list_of_essays)
    write_chunked_essay_to_file(dict_of_essays_with_chunks)
    # write_essaydict_to_file(dict_of_essays)
    end_time = time.time()
    print("time",end_time - start_time)
