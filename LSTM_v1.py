import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM,Lambda
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.layers import Bidirectional, Input
from keras.initializers import Constant
from keras.optimizers import Adam
from utilities import  *
from helper_functions import *
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import mean_squared_error
import time
from itertools import zip_longest


#constants:
embedding_dim = 300 # Len of vectors
max_features = 30000 # this is the number of words we care about
vocabulary_size = 5000
no_of_chunks = 3
# maybe???
sequence_length = 500



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


def run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels):
    model = Sequential()
    # model.add(Embedding(num_words, embedding_dim,embeddings_initializer=Constant(embedding_matrix),
    #                     input_length=sequence_length, trainable=True)) #bug check
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True),input_shape=(no_of_chunks,embedding_dim)))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(no_of_chunks, 400))) #average
    # model.add(Flatten())
    # ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Dense(200, input_dim=400, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate
    # Fit the model
    model.summary()
    model.fit(X_train, y_train, epochs=3, batch_size=2)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    # round predictions
    # rounded = [round(x[0]) for x in predictions]
    print(predictions)
    np.savetxt('prediction_output/test.out', predictions, delimiter='\n')
    print(y_test)
    print("RMSE", np.sqrt(mean_squared_error(y_test, predictions)))
    print("pearson", K.eval(correlation_coefficient_loss(y_test,predictions)))


def multiply_test(x_test, y_test):
    x_test_chunk = []
    y_test_chunk = []
    for i in range(len(x_test)):
        for count in range(no_of_chunks):
            x_test_chunk.append(x_test[i])
            y_test_chunk.append(y_test[i])

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
    # data = data.T # 1x 40
    # lets keep a couple of thousand samples back as a test set
    X_train, X_test, y_train, y_test = train_test_split(essays_with_avg_chunked, labels, test_size=0.2)
    print("test set size " + str(len(X_train)))
    # (10382, 5)
    # (2596, 5)
    # (10382,)
    # (2596,)

    X_train, X_test = np.array(X_train) , np.array(X_test)

    print(np.shape(X_train))
    print(np.shape(X_test))

    print(len(X_test))
    X_test, y_test = multiply_test(X_test, y_test)
    print(len(X_test))
    # print(np.shape(y_train))
    # print(np.shape(y_test))

    X_train = X_train[:10,:,:] # 10 x 3353
    y_train = y_train[:10]
    X_test = X_test[:10,:,:]
    y_test = y_test[:10]
    # print(y_test)
    num_words = len(word_index) + 1

    # hyperparameters:
    run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels)
    end = time.time()
    print("time", end - start)
