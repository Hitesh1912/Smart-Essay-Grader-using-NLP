from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utilities import  *
import time
from itertools import zip_longest




#constants:
embedding_dim = 300 # Len of vectors
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


def word_tokenize(data_arr, essay_data, sequence_length):
    # data = data.split(" ")
    # data = list(data)
    tokenizer = Tokenizer(num_words=vocabulary_size)
    print('fitting tokenizer on whole essay')
    tokenizer.fit_on_texts(essay_data)
    print('fitting complete')
    tokenized_essays = []
    tokenized_chunks = []
    # this takes our sentences and replaces each word with an integer
    for chunk_arr in data_arr:
        for chunk in chunk_arr:
            chunk = tokenizer.texts_to_sequences(chunk)
            print(tokenizer.sequences_to_texts(chunk))
            # we then pad the sequences so they're all the same length (sequence_length)
            # chunk = pad_sequences(chunk, 50, padding='post')  #check
            tokenized_chunks.append(chunk)
            # print(tokenized_chunks)
            # print(type(tokenized_chunks))
        # print(tokenized_chunks)
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


# def grouper(iterable, n, fillvalue=None):
#     args = [iter(iterable)] * n
#     return zip_longest(*args, fillvalue=fillvalue)

def chunks(sentences):
    sentences = np.array(sentences)
    return_arr = np.array_split(sentences, 3)
    # return_arr = np.array(return_arr)
    return return_arr


def avg_chunk_word_encoding(data,embedding_matrix):
    essays = []
    chunks_list = []
    temp_chunk_matrix = []
    for essay in data:
        for chunk in essay:
            for idx in chunk:
                temp_chunk_matrix.append(embedding_matrix[idx])
            chunks_list.append(np.mean(temp_chunk_matrix,axis=0))
        essays.append(chunks_list)
    print(np.shape(essays))
    return essays
