import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Lambda, GRU, SimpleRNN
from keras.layers import Bidirectional
from keras.optimizers import Adam
from helper_functions import *
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import mean_squared_error
import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.utils import shuffle

# constants:
embedding_dim = 300  # Len of vectors
max_features = 30000  # this is the number of words we care about
vocabulary_size = 5000
no_of_chunks = 2
sequence_length = 500
threshold = 0.3


# runs the LSTM model
def run_bi_directional_two_layer_lstm(X_train, y_train, X_test, y_test, num_words,embedding_matrix,sequence_length,labels, essay_ids):
    model = Sequential()
    # ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True),input_shape=(no_of_chunks,embedding_dim)))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    # model.add(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(no_of_chunks, 400))) #average

    # Feed forward neural network on essay encoding
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate
    # Fit the model
    # model.summary()
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    # np.savetxt('prediction_output/pred_test.out', predictions, delimiter='\n')
    # np.savetxt('prediction_output/real_test.out', y_test, delimiter='\n')
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, predictions)))
    print("pearson: ", pearsonr(predictions.reshape(np.shape(predictions)[0]), y_test))

    X_test_chunk, y_test_chunk, chunk_id = multiply_test(X_test, y_test, essay_ids)
    y_chunk_predictions = model.predict(X_test_chunk)
    for i in range(len(y_chunk_predictions)):
        if y_test_chunk[i] - y_chunk_predictions[i] >= threshold:
            essay_chunk_arr = chunk_id[i].split(',')
            print("The essay id " + essay_chunk_arr[0] + " and chunk number "+essay_chunk_arr[1] + " is weak")


# runs the GRU model
def run_gru(X_train, y_train, X_test, y_test, num_words,embedding_matrix,sequence_length,labels):
    model = Sequential()

    # ADD THE BI-DRECTIONAL GRU HIDDEN LAYER AS INPUT
    model.add(Bidirectional(GRU(200,dropout=0.2,recurrent_dropout=0.2, return_sequences=True),
                            input_shape=(no_of_chunks,embedding_dim)))
    model.add(Bidirectional(GRU(200,dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(no_of_chunks, 400))) # average to get essay encoding

    # Feed forward neural network on essay encoding
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate

    # Fit the model
    y_train = np.array(y_train)
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, predictions)))
    print("pearson: ", pearsonr(predictions.reshape(np.shape(predictions)[0]), y_test))


# runs the vanilla RNN model
def run_rnn(X_train, y_train, X_test, y_test, num_words,embedding_matrix,sequence_length,labels):
    model = Sequential()

    # ADD THE BI-DRECTIONAL RNN HIDDEN LAYER AS INPUT
    model.add(Bidirectional(SimpleRNN(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
                            input_shape=(no_of_chunks,embedding_dim)))
    model.add(Bidirectional(SimpleRNN(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(no_of_chunks, 400))) # average

    # Feed forward neural network on essay encoding
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate

    # Fit the model
    y_train = np.array(y_train)
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, predictions)))
    print("pearson: ", pearsonr(predictions.reshape(np.shape(predictions)[0]), y_test)[0])


# @input: test data
# @returns: chunked data
def multiply_test(x_test, y_test, essay_ids):
    x_test_chunk = []
    y_test_chunk = []
    chunk_ids = []
    for i in range(len(x_test)):
        chunk_no = 0
        for chunk in x_test[i]:
            temp_chunk_x = []
            for count in range(no_of_chunks):
                temp_chunk_x.append(np.array(chunk))
            y_test_chunk.append(y_test[i])
            x_test_chunk.append(np.array(temp_chunk_x))
            chunk_ids.append(str(essay_ids[i]) + ',' + str(chunk_no))
            chunk_no += 1
    return np.array(x_test_chunk), y_test_chunk, chunk_ids


def train_test_split_manual(training, labels, essay_ids):
    training, labels, essay_ids = shuffle(training, labels, essay_ids)
    per_80 = int(len(training) * 0.8)
    return training[:per_80], labels[:per_80], training[per_80:], labels[per_80:], essay_ids[per_80:]


if __name__ == '__main__':
    start = time.time()
    # step1: create embedding index from glove
    #=================================================================
    word2vec_data = importdata()  # read vector.txt
    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]
    print("glove vector:::embedding index", len(embedding_index))
    #=================================================================
    # load pretrained word2vec
    # filename = 'GoogleNews-vectors-negative300.bin'
    # model = KeyedVectors.load_word2vec_format(filename, binary=True)
    # print("loading google word2vec",model)
    # # create embedding index dictionary from pretrained word vectors
    # words = list(model.wv.vocab) #300000
    # embedding_index = {}
    # for word in words:
    #     coefs = np.asarray(model[word], dtype='float32')
    #     embedding_index[word] = coefs
    # embedding_index['<unk>'] = np.random.uniform(-1,1,(300,))
    # print('Found %s word vectors.' % len(embedding_index))
    print("embedding index generated")
    # ===================================================================

    training_data = open('dict_of_chunked_essays.txt', 'r').read()
    text_data = eval(training_data)  #bugg extra space need re processing
    essay_list = []
    sequence_list = []
    essay_data = ""  # corpus of all the essays 12978
    for essay_id in text_data:
        essay_list.append(chunks(text_data[essay_id]))
    print("essay set with fixed chunks",np.shape(essay_list))
    # np.savetxt("essay_list.txt",essay_list,delimiter=",",fmt="%s")

    # creating unigrams words for all essays text and essay list containing list of chunks
    # structure : list of essays :list of chunks :list of unigrams words
    essay_data = ''
    essay_ids = []
    for essay_id in text_data:
        text = " ".join(text_data[essay_id])
        essay_ids.append(essay_id)
        essay_data += text
    essay_data = essay_data.split(" ")  # into unigrams tokens before passing to token

    essay_list1 = []
    for essay in essay_list:
        chunks_list = []
        for chunk in essay:
            chunk = " ".join(chunk).strip()
            chunk = chunk.split(" ")
            chunks_list.append(chunk)
        essay_list1.append(chunks_list)
    print("essay list with fixed chunks", np.shape(essay_list1))
    # =================================================================
    # step3: convert word to ordered unique number tokens to form sequence
    data, tokenizer = word_tokenize(essay_list1, essay_data, sequence_length)
    print("essay list", np.shape(data))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    # num_words = min(max_features, len(word_index)) + 1
    # print(num_words)
    # =================================================================
    # step3: create initial embedding matrix using embedding index and word-representation i.e number as index in matrix
    embedding_matrix = create_embedding_matrix(word_index,embedding_index)
    print("ebedding matrix created",np.shape(embedding_matrix))  # 300001 x 200

    # average the words matrix of each chunk to 1 x vector_dim
    essays_with_avg_chunked = avg_chunk_word_encoding(data, embedding_matrix)

    # instead use this code to get list of scores
    with open('list_of_scores.txt', 'r') as fp:
        labels = [float(score.rstrip()) for score in fp.readlines()]
    # lets keep a couple of thousand samples back as a test set
    # X_train, X_test, y_train, y_test = train_test_split(essays_with_avg_chunked, labels, test_size=0.2, shuffle=False)
    X_train, y_train, X_test, y_test, essay_ids = train_test_split_manual(essays_with_avg_chunked, labels, essay_ids)
    X_train, X_test = np.array(X_train), np.array(X_test)

    num_words = len(word_index) + 1
    # models
    run_bi_directional_two_layer_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels, essay_ids)
    run_gru(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels)
    run_rnn(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels)
    end = time.time()
    print("time", end - start)
