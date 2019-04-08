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
from preprocessing import clean_text
from sklearn.model_selection import train_test_split
from keras import backend as K
import time


#constants:
embedding_dim = 200 # Len of vectors
max_features = 30000 # this is the number of words we care about
vocabulary_size = 5000


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
    # model.add(SpatialDropout1D(0.2))
    # model.add(Bidirectional(CuDNNLSTM(200, return_sequences=True)))
    # model.add(Bidirectional(CuDNNLSTM(32)))
    # model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Bidirectional(LSTM(200,dropout=0.2,recurrent_dropout=0.2,return_sequences=True)))
    model.add(Lambda(lambda x: K.mean(x, axis=1), input_shape=(sequence_length, 200)))
    # model.add(Flatten())
    # model.add(Dense(units=200, activation='softmax')) # # LSTM hidden layer -> FF INPUT

    # ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Dense(200, input_dim=200, activation='relu'))  # FF hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # learning rate
    print(model.summary())
    # Fit the model
    model.fit(X_train, y_train, epochs=2, batch_size=2)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X_test)
    # round predictions
    # rounded = [round(x[0]) for x in predictions]
    print(np.shape(predictions))
    print(predictions)
    print(y_test)
    print("MSE", mean_squared_error(y_test, predictions))
    print("pearson", K.eval(correlation_coefficient_loss(y_test,predictions)))

    #
    # print("train accuracy", accuracy_val(y, rounded))
    #
    # # calculate predictions
    # predictions_t = model.predict(X_test)
    # # round predictions
    # rounded_t = [round(x[0]) for x in predictions_t]
    # print(rounded_t)
    # # print("MSE",mean_squared_error(y,rounded))
    # print("test accuracy", accuracy_val(y_test, rounded_t))

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    # print(model.summary())
    #
    # # model.evaluate(data,labels)
    #
    # # batch_size = 128
    # model.fit(data, labels, epochs=5, verbose=1, validation_split=0.1)
    #
    # model.predict(data)
    # train LSTM



def word_tokenize(data,sequence_length):
    # data = data.split(" ")
    # data = list(data)
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(data)

    # this takes our sentences and replaces each word with an integer
    X = tokenizer.texts_to_sequences(data)
    # print(tokenizer.sequences_to_texts(X))
    # we then pad the sequences so they're all the same length (sequence_length)
    X = pad_sequences(X, sequence_length)  #check
    return X, tokenizer



def create_embedding_matrix(word_index,embeddings_index):
    num_words = min(max_features, len(word_index)) + 1
    print(num_words)
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


if __name__ == '__main__':
    start = time.time()
    # text = "Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listenin"
    # step1: create embedding index from glove
    word2vec_data = importdata()  #read vector.txt
    embedding_index = {}
    for row in word2vec_data:
        embedding_index[row[0]] = row[1:]

    print("glove vector:::embedding index", len(embedding_index))

    training_data = open('dict_of_chunked_essays.txt', 'r').read()
    text_data = eval(training_data)
    essay_list = []
    sequence_list = []
    essay_data = ""  #corpus of all the essays 12978
    max_len = 0
    for essay_id in text_data:
        text = " ".join(text_data[essay_id])
        # text = clean_text(text)
        max_len = len(text) if len(text) > max_len else max_len
        essay_data = essay_data + text
        essay_list.append(text)
        # print(text)
        # calling from preprocessing.py
    print("max_len",max_len)
    sequence_length = max_len  # max length of an essay

    #step2: convert word to ordered unique number tokens to form sequence
    data, tokenizer = word_tokenize(essay_list,sequence_length)

    print("data",np.shape(data))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    num_words = min(max_features, len(word_index)) + 1
    print(num_words)

    #step3: create initial embedding matrix using embedding index and word-representation i.e number as index in matrix
    embedding_matrix = create_embedding_matrix(word_index,embedding_index)
    #
    print(np.shape(embedding_matrix))  #31 x 200

    # labels = eval(open('list_of_scores.txt', 'r').read())

    #instead above line use below code to get list of scores
    with open('list_of_scores.txt', 'r') as fp:
        labels= [float(score.rstrip()) for score in fp.readlines()]
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
