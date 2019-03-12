# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.layers import Bidirectional,CuDNNLSTM
from keras.initializers import Constant

import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re



# Function importing Dataset
def importdata():
    # train_data = pd.read_csv(
    #     'training_set_rel3.tsv', skipinitialspace=True, header=None)
    data = pd.read_table('word_embedding_data', header=None, sep = " ")
    # print(data.head())
    print(data.shape)
    return data.values


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



def run_lstm(data,num_words,embedding_matrix):
    embedding_dim = 50 # len of vectors
    max_features = 2000
    sequence_length = 52

    ## Network architecture
    labels = np.random.randint(0,1,size=(np.shape(data)[0],1))

    model = Sequential()
    # model.add(Embedding(20000, 100, input_length=50))
    #
    #
    #
    # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # # model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
    #
    # model.add(Dense(1, activation='sigmoid'))
    # # model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    # ## Fit the model
    # model.fit(data, np.array(labels), validation_split=0.4, epochs=3)

    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=sequence_length,
                        trainable=True))

    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # model.evaluate(data,labels)

    # batch_size = 128
    model.fit(data, labels, epochs=5, verbose=1, validation_split=0.1)

    model.predict(data)
    # train LSTM



def word_tokenize(data):
    max_features = 20000  # this is the number of words we care about
    sequence_length = 52


    # Create sequence
    # vocabulary_size = 20000
    # tokenizer = Tokenizer(num_words= vocabulary_size)
    # tokenizer.fit_on_texts(parsed_text)
    # sequences = tokenizer.texts_to_sequences(parsed_text)
    # data = pad_sequences(sequences, maxlen=50)


    tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>', filters=' ')
    tokenizer.fit_on_texts(data)

    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(data)

    # we then pad the sequences so they're all the same length (sequence_length)
    X = pad_sequences(X, sequence_length)

    # lets keep a couple of thousand samples back as a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # print("test set size " + str(len(X_test)))
    return X, tokenizer

def create_embedding_matrix(word_index,embeddings_index):
    num_words = min(max_features, len(word_index)) + 1
    print(num_words)

    embedding_dim = 50

    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # for each word in out tokenizer lets try to find that work in our w2v model
    for word, i in word_index.items():
        if i > max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # we found the word - add that words vector to the matrix
            embedding_matrix[i] = embedding_vector
        else:
            # doesn't exist, assign a random vector
            embedding_matrix[i] = np.random.randn(embedding_dim)

    return embedding_matrix


if __name__ == '__main__':

    text = "Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listenin"
    parsed_text = clean_text(text)

    data, tokenizer = word_tokenize(parsed_text)

    print("data",np.shape(data))

    embedding_dim = 50
    max_features = 20000
    num_words = len(data)
    word2vec_data = importdata()
    # embedding_matrix = word2vec_data[:, 1:np.shape(word2vec_data)[1]]
    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]

    # print("embedding index",embedding_index)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    num_words = min(max_features, len(word_index)) + 1
    # print(num_words)

    embedding_matrix = create_embedding_matrix(word_index,embedding_index)

    # print(np.shape(embedding_matrix))  #31 x 50

    run_lstm(data,num_words,embedding_matrix)
