import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
import time
from spellchecker import SpellChecker
from sklearn import preprocessing


# to use spellchecker
spell = SpellChecker()


# Function importing Dataset from the training file
def importdata():
    data = pd.read_table('training_set_rel3.tsv', header=None, sep = "\t+", encoding='iso-8859-1')
    return data.values


# Function to split the dataset into x and y
# where x includes essay id, essay set and essay
# and y includes the scores
def splitdataset(data):
    x = data[1:, 0:3]
    y = data[1:, 3:np.shape(data)[1]]
    return x, y


# This function is used to create essay scores list
# for each of the essay sets
# As each essay set had separate scoring criterion
# we had to normalize each essay set individually
def create_score_list(essay_set, y_train):
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
    return list_y_score


# This function picks essay from the array
# and appends to a list which is then
# returned
def preprocess_data(training_data):
    #split data till essay column
    essay = []
    for row in training_data:
        essay.append(row[2])
    return essay


# This function creates a dictionary of essay
# which has essays represented as a list of chunks
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
            clean_sentence = clean_text(sentence)
            list_of_chunks.append(clean_sentence)
        i = 0
        length = len(list_of_chunks)
        while i < length:
            if list_of_chunks[i] == '':
                list_of_chunks.remove(list_of_chunks[i])
                length = length - 1
                continue
            i = i + 1
        dict_of_essays_with_chunks[essay_counter] = list_of_chunks
        essay_counter += 1
    return dict_of_essays_with_chunks


# This function creates a dictionary of essays
def create_dict_of_essays(x_train):
    counter = 1
    dict_of_essays = dict()
    for row in x_train:
        cleaned_essay = clean_text(row[2])
        dict_of_essays[counter] = cleaned_essay
        counter += 1
    return dict_of_essays


# This function performs the cleaning of the text
# Removes stopwords, punctuations, numerical values,
# converts to lowercase and handles some abbreviations
def clean_text(text):
    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
    text = text.replace('"', "")
    text = text.replace('.', "")
    text = text.replace('?', '')
    text = text.replace("!", "")

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
    text = text.replace(" - ", " ")
    text = text.replace("  ", " ")
    return text


# This function is used for spelling correction
def correct_spelling(sentence):
    words = sentence.split(" ")
    # misspelled = spell.unknown(words)
    known_words = spell.known(words)
    unknown_words = spell.unknown(words)
    final_list = list()
    for word in words:
        if word in known_words:
            final_list.append(word)
        if word in unknown_words:
            final_list.append(spell.correction(word))
    sentence = ' '.join(final_list)
    return sentence


# This function creates the file of dictionary of essay with chunks
def write_chunked_essay_to_file(dict_of_essays_with_chunks):
    wf = open("dict_of_chunked_essays.txt", "w", encoding='iso-8859-1')
    wf.write(str(dict_of_essays_with_chunks))


# This function creates the file of list of normalized scores
def write_scorelist_to_file(list_of_scores):
    np.savetxt('list_of_scores.txt', list_of_scores, delimiter=',', fmt='%1.3f')


# This function creates the file of dictionary of essay
def write_essaydict_to_file(dict_of_essays):
    wf = open("dict_of_essays.txt", "w", encoding='iso-8859-1')
    wf.write(str(dict_of_essays))


if __name__ == '__main__':
    start_time = time.time()

    dict_of_scores = dict()
    dict_of_essays = dict()

    training_data = importdata()
    x_train, y_train = splitdataset(training_data)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = y_train[:,2]
    essay_set = x_train[:, 1]
    essay_set = essay_set.astype(float)

    list_y_score = list()
    list_y_score = create_score_list(essay_set, y_train)
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

    list_of_essays = preprocess_data(training_data)
    dict_of_essays_with_chunks = create_essay_chunks(list_of_essays)
    dict_of_essays = create_dict_of_essays(x_train)
    write_chunked_essay_to_file(dict_of_essays_with_chunks)
    write_scorelist_to_file(y_train)
    write_essaydict_to_file(dict_of_essays)
    end_time = time.time()
    print("time",end_time - start_time)
