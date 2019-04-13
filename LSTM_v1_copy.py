from keras.models import Sequential
from keras.layers import Dense, LSTM,Lambda
from keras.layers import Bidirectional
from keras.optimizers import Adam
from helper_functions import *
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import mean_squared_error
import time
import numpy as np
from scipy.stats import pearsonr


#constants:
embedding_dim = 300 # Len of vectors
max_features = 30000 # this is the number of words we care about
vocabulary_size = 5000
no_of_chunks = 3
sequence_length = 500



def run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels):
    model = Sequential()
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True),input_shape=(no_of_chunks,embedding_dim)))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(no_of_chunks, 400))) #average
    # ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200, activation='relu'))  # FF hidden layer
    model.add(Dense(200,activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate
    # Fit the model
    model.summary()
    model.fit(X_train, y_train, epochs=20, batch_size=128)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    print(predictions)
    # np.savetxt('prediction_output/test.out', predictions, delimiter='\n') #for predicting essay score
    np.savetxt('prediction_output/test_chunks.out', predictions, delimiter='\n') #for predicting chunk score
    print(y_test)
    print("RMSE", np.sqrt(mean_squared_error(y_test, predictions)))
    print("pearson", pearsonr(predictions.reshape((np.shape(predictions)[0])), y_test))


def multiply_test(x_test, y_test):
    x_test_chunk = []
    y_test_chunk = []
    for i in range(len(x_test)):
        for chunk in x_test[i]:
            temp_chunk_x = []
            for count in range(no_of_chunks):
                temp_chunk_x.append(np.array(chunk))
            y_test_chunk.append(y_test[i])
            x_test_chunk.append(temp_chunk_x)
    return np.array(x_test_chunk), np.array(y_test_chunk)


if __name__ == '__main__':
    start = time.time()
    # step1: create embedding index from glove
    #=================================================================
    word2vec_data = importdata()  # read vector.txt
    # print(word2vec_data[0:,0])
    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]
    print("glove vector:::embedding index", len(embedding_index))
    #=================================================================

    training_data = open('dict_of_chunked_essays3.txt', 'r').read()
    text_data = eval(training_data)

    essay_list = []
    sequence_list = []
    essay_id_list = []
    chunked_essay_dict = {}
    essay_data = ""  # corpus of all the essays 12978
    for essay_id in text_data:
        essay_id_list.append(essay_id)
        chunked_data = chunks(text_data[essay_id])
        chunked_essay_dict[essay_id] = chunked_data
        essay_list.append(chunked_data)
    print("essay set with fixed chunks",np.shape(essay_list))
    wf = open("chunked_essays_dict.txt", "w", encoding='iso-8859-1')
    wf.write(str(chunked_essay_dict))


    # creating unigrams words for all essays text and essay list containing list of chunks
    # structure : list of essays :list of chunks :list of unigrams words
    essay_data = ''
    for essay_id in text_data:
        text = " ".join(text_data[essay_id])
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
    print(np.shape(embedding_matrix))  #300001 x 200

    #average the words matrix of each chunk to 1 x vector_dim
    essays_with_avg_chunked = avg_chunk_word_encoding(data, embedding_matrix)
    print(type(essays_with_avg_chunked))
    # print(essays_with_avg_chunked[0])

    #instead use this code to get list of scores
    with open('list_of_scores3.txt', 'r') as fp:
        labels = [float(score.rstrip()) for score in fp.readlines()]
    # print(labels)

    #passing essay_id in the cross validation split to track the shuffled essay for analysis purpose
    essay_id_list = np.array(essay_id_list)
    essay_id_list = np.expand_dims(essay_id_list, axis=1)
    print("essay id",np.shape(essay_id_list))

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(essays_with_avg_chunked, labels, essay_id_list, test_size=0.2)
    # (10382, 5)
    # (2596, 5)

    #for running on full training set
    # X_train = essays_with_avg_chunked
    # y_train = labels
    # X_test = essays_with_avg_chunked
    # y_test = labels

    #refer this file as essay index lookup
    np.savetxt("prediction_output/test_essay_index.txt", z_test, fmt="%i")

    X_train, X_test = np.array(X_train), np.array(X_test)
    X_test, y_test = multiply_test(X_test, y_test) #for chunks uncomment it!
    print(len(X_test))

    print("x_train", np.shape(X_train))
    print("x_test", np.shape(X_test))
    print("y_test", np.shape(y_test))
    print("y_train",np.shape(y_train))
    print("train_idx",np.shape(z_train))
    print("test_idx",np.shape(z_test))



    X_train = X_train[:,:,:]
    y_train = y_train[:]
    X_test = X_test[:,:,:]
    y_test = y_test[:]
    # print(y_test)
    num_words = len(word_index) + 1

    # hyperparameters: ?

    run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels)
    end = time.time()
    print("time", end - start)
