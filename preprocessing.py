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
    data = pd.read_table('training_set_rel3.tsv', header=None, sep = "\t+", encoding='iso-8859-1')
    # print(data.head())
    # print(data.shape)
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
    return dict_of_essays_with_chunks


def create_list_of_scores(y_train):
    list_of_scores = list()
    for row in y_train:
        list_of_scores.append(row[2])
    return list_of_scores


def create_dict_of_essays(x_train):
    counter = 1
    dict_of_essays = dict()
    for row in x_train:
        cleaned_essay = clean_text(row[2])
        dict_of_essays[counter] = cleaned_essay
        counter += 1
    return dict_of_essays


def write_chunked_essay_to_file(dict_of_essays_with_chunks):
    wf = open("dict_of_chunked_essays3.txt", "w", encoding='iso-8859-1')
    wf.write(str(dict_of_essays_with_chunks))


def write_scorelist_to_file(list_of_scores):
    wf = open("list_of_scores3.txt", "w", encoding='iso-8859-1')
    np.savetxt('list_of_scores3.txt', list_of_scores, delimiter=',', fmt='%1.3f')
    # wf.write(str(list_of_scores))


def write_essaydict_to_file(dict_of_essays):
    wf = open("dict_of_essays3.txt", "w", encoding='iso-8859-1')
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
    # text = re.sub(r"@ORGANIZATION[0-9]", "organization", text)
    # text = re.sub(r"@CAPS[0-9]", "name", text)
    # text = re.sub(r"@NUM[0-9]", "number", text)
    # text = re.sub(r"@LOCATION[0-9]", "location", text)
    # text = re.sub(r"@DATE[0-9]", "date", text)
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
    y_train = np.array(y_train)
    y_train = y_train[:,2]
    # print(y_train)
    # exit()
    essay_set = x_train[:, 1]
    essay_set = essay_set.astype(float)
    idx_1 = np.array(np.where(essay_set == 1.0)[0])
    idx_2 = np.array(np.where(essay_set == 2.0)[0])
    idx_3 = np.array(np.where(essay_set == 3.0)[0])
    idx_4 = np.array(np.where(essay_set == 4.0)[0])
    idx_5 = np.array(np.where(essay_set == 5.0)[0])
    idx_6 = np.array(np.where(essay_set == 6.0)[0])
    idx_7 = np.array(np.where(essay_set == 7.0)[0])
    idx_8 = np.array(np.where(essay_set == 8.0)[0])

    list_y_score = list()
    idx_1 = idx_1.reshape((np.shape(idx_1)[0]))
    y_score1 = y_train[idx_1]
    list_y_score.append(y_score1)
    idx_2 = idx_2.reshape((np.shape(idx_2)[0]))
    y_score2 = y_train[idx_2]
    list_y_score.append(y_score2)
    idx_3 = idx_3.reshape((np.shape(idx_3)[0]))
    y_score3 = y_train[idx_3]
    list_y_score.append(y_score3)
    idx_4 = idx_4.reshape((np.shape(idx_4)[0]))
    y_score4 = y_train[idx_4]
    list_y_score.append(y_score4)
    idx_5 = idx_5.reshape((np.shape(idx_5)[0]))
    y_score5 = y_train[idx_5]
    list_y_score.append(y_score5)
    idx_6 = idx_6.reshape((np.shape(idx_6)[0]))
    y_score6 = y_train[idx_6]
    list_y_score.append(y_score6)
    idx_7 = idx_7.reshape((np.shape(idx_7)[0]))
    y_score7 = y_train[idx_7]
    list_y_score.append(y_score7)
    idx_8 = idx_8.reshape((np.shape(idx_8)[0]))
    y_score8 = y_train[idx_8]
    list_y_score.append(y_score8)

    normalized_score_list = list()
    for y_score in list_y_score:
        y_score = y_score.reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler()
        normalizer = scaler.fit(y_score)
        normalized_score = scaler.transform(y_score)
        normalized_score_list.append(normalized_score)

    normalized_score_list = np.array(normalized_score_list)
    y_train = np.concatenate(normalized_score_list, axis=0)
    y_train = y_train.reshape(np.shape(y_train)[0])
    # scaler = preprocessing.MinMaxScaler()
    # normalizer = scaler.fit(y_train)
    # normalized_score = scaler.transform(y_train)
    # normalize all the scores in y_train
    # normalizer = preprocessing.StandardScaler()
    # normalized_score = normalizer.fit_transform(y_train)

    essays = x_train[:, 2]
    # print(essays[:3])
    counter = 1
    processed_essays = []

    for essay in essays:
        processed_essays.append(clean_text(essay))

    for essay in processed_essays:
        dict_of_essays[counter] = essay
        counter = counter + 1

    list_of_essays = preprocess_data(training_data)
    dict_of_essays_with_chunks = create_essay_chunks(list_of_essays)
    # list_of_scores = create_list_of_scores(y_train)
    dict_of_essays = create_dict_of_essays(x_train)
    # print(dict_of_essays)
    write_chunked_essay_to_file(dict_of_essays_with_chunks)
    # write_scorelist_to_file(y_train)
    write_essaydict_to_file(dict_of_essays)
    end_time = time.time()
    print("time",end_time - start_time)
