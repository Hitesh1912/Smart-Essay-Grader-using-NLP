from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utilities import  *
import time
from itertools import zip_longest
from nltk import ngrams



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
    print(data.shape)
    return data.values


# def word_tokenize(data_arr, essay_data, sequence_length):
#     # data = data.split(" ")
#     # data = list(data)
#     tokenizer = Tokenizer(num_words=vocabulary_size)
#     print('fitting tokenizer on whole essay')
#     tokenizer.fit_on_texts(essay_data)
#     print('fitting complete')
#     tokenized_essays = []
#     tokenized_chunks = []
#     # this takes our sentences and replaces each word with an integer
#     for chunk_arr in data_arr:
#         for chunk in chunk_arr:
#
#             chunk  = " ".join(chunk)
#             # print(np.shape(chunk), chunk)
#             chunk = chunk.split(" ")
#             chunk = tokenizer.texts_to_sequences(chunk)
#             print(tokenizer.sequences_to_texts(chunk))
#             # we then pad the sequences so they're all the same length (sequence_length)
#             # chunk = pad_sequences(chunk, 50, padding='post')  #check
#             tokenized_chunks.append(chunk)
#             # print(tokenized_chunks)
#             # print(type(tokenized_chunks))
#         # print(tokenized_chunks)
#         tokenized_essays.append(tokenized_chunks)
#     return tokenized_essays, tokenizer


def word_tokenize(essay_list, essay_data, sequence_length):
    tokenizer = Tokenizer(num_words=vocabulary_size)
    print('fitting tokenizer on whole essay')
    tokenizer.fit_on_texts(essay_data)
    print('fitting complete')
    tokenized_essays = []
    # this takes our sentences and replaces each word with an integer
    for essay in essay_list:
        tokenized_chunks = []
        for chunk in essay:
            chunk_seq = tokenizer.texts_to_sequences(chunk)
            # print(chunk_seq)
            # print(tokenizer.sequences_to_texts(chunk_seq))
            # we then pad the sequences so they're all the same length (sequence_length)
            # chunk = pad_sequences(chunk, 50, padding='post')  #check
            tokenized_chunks.append(chunk_seq)
        tokenized_essays.append(tokenized_chunks)
    return tokenized_essays, tokenizer



def create_embedding_matrix(word_index,embeddings_index):
    num_words = min(max_features, len(word_index)) + 1
    # first create a matrix of zeros, this is our embedding matrix
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
    # for each word in out tokenizer lets try to find that work in our w2v model
    for word, i in word_index.items():
        # if i > max_features:
        #     continue
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


def avg_chunk_word_encoding(essays,embedding_matrix):
    essays_list = []
    for essay in essays:
        chunks_list = []
        for chunk in essay:
            temp_word_matrix = []
            if chunk:
                for word_idx in chunk:
                    if word_idx in embedding_matrix:
                        temp_word_matrix.append(embedding_matrix[word_idx])
                print("word matrix", np.shape(temp_word_matrix), len(chunk))
            else:
                print(chunk)
            avg = np.mean(temp_word_matrix,axis=0)
            print(np.shape(avg))
            chunks_list.append(avg)
        essays_list.append(chunks_list)
    print("essay encoded",np.shape(essays_list))
    return essays_list


def clean_text(text):
    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)
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
    text = re.sub(r"@ORGANIZATION[0-9]", "organization", text)
    text = re.sub(r"@CAPS[0-9]", "name", text)
    text = re.sub(r"@NUM[0-9]", "number", text)
    text = re.sub(r"@LOCATION[0-9]", "location", text)
    text = re.sub(r"@DATE[0-9]", "date", text)
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

    return text