import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM,Lambda
from keras.layers.embeddings import Embedding
from keras.layers import TimeDistributed
from keras.layers import Bidirectional,CuDNNLSTM
from keras.initializers import Constant
from keras.optimizers import Adam
from utilities import  *
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import mean_squared_error
import time
from itertools import zip_longest


#constants:
embedding_dim = 200 # Len of vectors
max_features = 30000 # this is the number of words we care about
vocabulary_size = 5000
no_of_chunks = 3

# maybe???
sequence_length = 500


# Function importing Dataset
def importdata():
    # train_data = pd.read_csv(
    #     'training_set_rel3.tsv', skipinitialspace=True, header=None)
    data = pd.read_table('vectors.txt', header=None, sep = " ")
    # print(data.head())
    print(data.shape)
    return data.values


# def mean_squared_error(actual, predicted):
#     mse = (np.square(np.array(actual) - np.array(predicted))).mean()
#     return mse


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
    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=sequence_length,
                        trainable=True))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(sequence_length, 200)))
    # model.add(Flatten())
    # model.add(Dense(units=200, activation='softmax')) # # LSTM hidden layer -> FF INPUT

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


def word_tokenize(data_arr, essay_data, sequence_length):
    # data = data.split(" ")
    # data = list(data)

    tokenizer = Tokenizer(num_words=vocabulary_size)
    print('here')
    tokenizer.fit_on_texts(essay_data)
    print('here')
    tokenized_essays = []
    tokenized_chunks = []
    # this takes our sentences and replaces each word with an integer
    for chunk_arr in data_arr:
        for chunk in chunk_arr:
            chunk = tokenizer.texts_to_sequences(chunk)
            # print(tokenizer.sequences_to_texts(X))
            # we then pad the sequences so they're all the same length (sequence_length)
            chunk = pad_sequences(chunk, 50, padding='post')  #check
            tokenized_chunks.append(chunk)
            print(tokenized_chunks)
            print(type(tokenized_chunks))
        tokenized_essays.append(tokenized_chunks)
    return tokenized_essays, tokenizer


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
            # doesn't exist, assign a random vector
            embedding_matrix[i] = embeddings_index.get('<unk>')
    return embedding_matrix


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def chunks(sentences):
    sentences = np.array(sentences)
    return_arr = np.array_split(sentences, 3)
    return return_arr


if __name__ == '__main__':
    start = time.time()
    # step1: create embedding index from glove
    word2vec_data = importdata()  # read vector.txt
    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]

    print("glove vector:::embedding index", len(embedding_index))

    training_data = open('dict_of_chunked_essays.txt', 'r').read()
    text_data = eval(training_data)
    essay_list = []
    sequence_list = []
    essay_data = ""  # corpus of all the essays 12978
    max_len = 0
    for essay_id in text_data:
        essay_list.append(chunks(text_data[essay_id]))
        # chunk_coding_arr = []
        # for text_zero in chunk_arr:
        #     text = text_zero[0]
        #
        #     # text = " ".join(text)
        #     # max_len = len(text.split(' ')) if len(text.split(' ')) > max_len else max_len
        #     # essay_data = essay_data + text
        #     chunk_coding_arr.append(text)
        # essay_list.append(chunk_coding_arr)
    # print("max_len",max_len)
    # sequence_length = max_len  # max length of an essay

    #step2: convert word to ordered unique number tokens to form sequence
    essay_data = ''
    for essay_id in text_data:
        text = " ".join(text_data[essay_id])
        essay_data += text
    data, tokenizer = word_tokenize(essay_list, essay_data, sequence_length)
    print("data",np.shape(data))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    num_words = min(max_features, len(word_index)) + 1
    print(num_words)
    exit()
    #step3: create initial embedding matrix using embedding index and word-representation i.e number as index in matrix
    embedding_matrix = create_embedding_matrix(word_index,embedding_index)
    #
    print(np.shape(embedding_matrix))  #31 x 200

    # labels = eval(open('list_of_scores.txt', 'r').read())

    #instead above line use below code to get list of scores
    with open('list_of_scores.txt', 'r') as fp:
        labels = [float(score.rstrip()) for score in fp.readlines()]
    # print(labels)

    # data = data.T # 1x 40

    # lets keep a couple of thousand samples back as a test set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
    print("test set size " + str(len(X_train)))

    X_train = X_train[:10,:] # 10 x 3353
    y_train = y_train[:10]
    X_test = X_test[:10,:]
    y_test = y_test[:10]
    print(y_test)

    run_lstm(X_train,y_train,X_test,y_test,num_words,embedding_matrix,sequence_length,labels)
    end = time.time()
    print("time", end - start)
