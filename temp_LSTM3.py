# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.layers import Bidirectional,CuDNNLSTM
from keras.initializers import Constant
from keras.optimizers import Adam
from utilities import  *
from preprocessing import clean_text
import numpy as np
import pandas as pd



#constants:
embedding_dim = 200 # Len of vectors
max_features = 20000 # this is the number of words we care about




# Function importing Dataset
def importdata():
    # train_data = pd.read_csv(
    #     'training_set_rel3.tsv', skipinitialspace=True, header=None)
    data = pd.read_table('vectors.txt', header=None, sep = " ")
    # print(data.head())
    print(data.shape)
    return data.values



def mean_squared_error(actual, predicted):
    mse = (np.square(np.array(actual) - np.array(predicted))).mean()
    return mse


def word_tokenize(data):
    # tokenizer = Tokenizer(num_words=237, split=' ', oov_token='<unw>', filters=' ')
    data = data.split(" ")
    sequence_length = 1  # testing for 1 sentence
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(data)
    # print(tokenizer.sequences_to_texts(X))
    # we then pad the sequences so they're all the same length (sequence_length)
    X = pad_sequences(X, sequence_length)  #check

    # lets keep a couple of thousand samples back as a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # print("test set size " + str(len(X_test)))
    return X, tokenizer


def create_embedding_matrix(word_index,embeddings_index):
    num_words = min(max_features, len(word_index)) + 1
    # print(num_words)

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
            embedding_matrix[i] = embeddings_index.get('<unk>')

    return embedding_matrix


def run_lstm(text_data):

    # delete later
    text_data1 = text_data
    temp_dic={}
    count = 0
    for essay_id in text_data1:
        temp_dic[essay_id] = text_data1[essay_id]
        count += 1
        if count == 10:
            break
    text_data = temp_dic
    count = 0
    for essay_id in text_data:
        input_embedding_matrix_arr = []
        output = []
        line_arr = text_data[essay_id]
        for text in line_arr:
            parsed_text = clean_text(text)  # calling from preprocessing.py
            data, tokenizer = word_tokenize(parsed_text)
            word_index = tokenizer.word_index
            num_words = min(max_features, len(word_index)) + 1
            data = data.T  # 1x 40
            embedding_matrix = create_embedding_matrix(word_index, embedding_index)

            input_embedding_matrix_arr.append(embedding_matrix)
            output.append(np.ones((1, 1)) * 0.6)

            sequence_length = 40

            ## Network architecture
            labels = np.random.randint(0, 1, size=(np.shape(data)[0], 1))

            model = Sequential()
            model.add(Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=sequence_length,
                                trainable=True))
        count += 1
        print(count)

        model.add(Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(units=200, activation='softmax')) # # LSTM hidden layer -> FF INPUT

    # ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Dense(200, input_dim=200, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer
    print('rere')

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate

    # Fit the model
    model.fit(data, output, epochs=100, batch_size=50)
    # # model.fit(data, y, epochs=100, batch_size=50)
    # print("training complete...")
    #
    # # calculate predictions
    # predictions = model.predict(data)
    # print("MSE",mean_squared_error(y,predictions))


if __name__ == '__main__':

    training_data = open('dict_of_chunked_essays.txt', 'r').read()
    text_data = eval(training_data)

    word2vec_data = importdata()  #read vector.txt

    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]

    print("embedding index",len(embedding_index))

    run_lstm(text_data)
