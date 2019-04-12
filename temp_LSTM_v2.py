import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM,Lambda
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.initializers import Constant
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import mean_squared_error
import time


#constants:
embedding_dim = 200 # Len of vectors
# max_features = 30000 # this is the number of words we care about
vocabulary_size = 5000


# Function importing Dataset
def importdata():
    data = pd.read_table('vectors.txt', header=None, sep = " ")
    # print(data.head())
    print(data.shape)
    return data.values


def correlation_coefficient_loss(y_true, y_pred):
    x = K.variable(np.array(y_true, dtype="float32"))
    y = K.variable(np.array(y_pred, dtype="float32"))
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length):

    #hyperparameters
    lstm_size = 200
    ffnn_size = 200
    epoch = 10
    learning_rate = 0.001
    bat_size = 32
    train_flag = True


    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=sequence_length,
                        trainable=train_flag))
    model.add(Bidirectional(LSTM(lstm_size,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_size,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(sequence_length, 400)))
    # model.add(Flatten())

    # ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Dense(ffnn_size, input_dim=400, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])  # learning rate
    # Fit the model
    model.summary()
    model.fit(X_train, y_train, epochs=epoch, batch_size=bat_size)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    print(predictions)
    # np.savetxt('prediction_output/test.out', predictions, delimiter='\n')
    print(y_test)
    print("MSE",mean_squared_error(y_test, predictions))
    print("RMSE", np.sqrt(mean_squared_error(y_test, predictions)))
    print("pearson", K.eval(correlation_coefficient_loss(y_test,predictions)))


def word_tokenize(data,sequence_length):
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(data)
    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(data)
    # we then pad the sequences so they're all the same length (sequence_length)
    X = pad_sequences(X, sequence_length, padding='post')  #check
    return X, tokenizer


def create_embedding_matrix(word_index,embeddings_index):
    num_words = min(max_features, len(word_index)) + 1
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
            embedding_matrix[i] = embeddings_index.get('<unk>')
    return embedding_matrix



def processing_dataset(data_set):
    text_data = eval(data_set)
    essay_list = []
    essay_data = ""  # corpus of all the essays 12978
    max_len = 0
    for essay_id in text_data:
        text = text_data[essay_id].split(' ')
        temp_text = []
        for val in text:
            if val:
                temp_text.append(val)
        text = temp_text
        text = " ".join(text)
        max_len = len(text.split(' ')) if len(text.split(' ')) > max_len else max_len
        essay_data = essay_data + text
        essay_list.append(text)
    print("max_len", max_len)
    sequence_length = max_len  # max length of an essay

    # step2: convert word to ordered unique number tokens to form sequence
    data, tokenizer = word_tokenize(essay_list, sequence_length)
    print("data after tokenizing", np.shape(data))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    # num_words = min(max_features, len(word_index)) + 1
    # print(num_words)
    # step3: create initial embedding matrix using embedding index and word-representation i.e number as index in matrix
    embedding_matrix = create_embedding_matrix(word_index, embedding_index)
    print("embedding matrix created",np.shape(embedding_matrix))  # 31 x 200
    return data, sequence_length, num_words, embedding_matrix





if __name__ == '__main__':
    start = time.time()
    # step1: create embedding index from glove
    word2vec_data = importdata()  #read vector.txt
    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]
    print("glove vector:::embedding index", len(embedding_index))

    training_data = open('dict_of_essays3.txt', 'r').read()
    data, sequence_length, num_words, embedding_matrix = processing_dataset(training_data)


    #instead above line use below code to get list of scores
    with open('list_of_scores3.txt', 'r') as fp:
        labels= [float(score.rstrip()) for score in fp.readlines()]

    #cross-validation split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    print(np.shape(X_train), np.shape(y_train))
    print(np.shape(X_test),np.shape(y_train))

    X_train = X_train[:1000,:] # 10 x 3353
    y_train = y_train[:1000]
    X_test = X_test[:200,:]
    y_test = y_test[:200]
    print("y_true")
    print(y_test)

    #processing test data
    # test_data = open('dict_of_essays_test.txt','r').read()
    # X_test, sequence_length, num_words, embedding_matrix = processing_dataset(test_data)
    #
    # with open('list_of_scores3.txt', 'r') as fp:
    #     y_test= [float(score.rstrip()) for score in fp.readlines()]

    #run on training and test on validation set

    #NOTE: INCCREASE THE BATCH SIZE WHEN YOU INCREASE THE DATA POINTS
    run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length)
    end = time.time()
    print("time", end - start)
