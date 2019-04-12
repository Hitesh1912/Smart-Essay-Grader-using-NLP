from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utilities import  *
import time
from itertools import zip_longest
from nltk import ngrams



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
    print(data.shape)
    return data.values


def word_tokenize(essay_list, essay_data, sequence_length):
    tokenizer = Tokenizer(num_words=vocabulary_size)
    print('fitting tokenizer on whole essay')
    # print(len(essay_data))
    # print(len(essay_data[0]))
    # print(essay_data[0])
    # exit()
    tokenizer.fit_on_texts(essay_data)
    print('fitting complete')
    tokenized_essays = []
    word_index_tok = tokenizer.word_index
    # this takes our sentences and replaces each word with an integer
    count = 0
    for essay in essay_list:
        tokenized_chunks = []
        for chunk in essay:
            # chunk_seq = tokenizer.texts_to_sequences(chunk)
            chunk_seq = []
            for word in chunk:
                if word in word_index_tok:
                    chunk_seq.append([word_index_tok[word]])
                else:
                    chunk_seq.append([word_index_tok['unk']])
            # tokenizer.
            chunk_seq = [x for x in chunk_seq if x!= []]  #to remove empty in sequence
            # print(chunk_seq)
            # print(tokenizer.sequences_to_texts(chunk_seq))
            # we then pad the sequences so they're all the same length (sequence_length)
            # chunk = pad_sequences(chunk, 50, padding='post')  #check
            tokenized_chunks.append(chunk_seq)
        tokenized_essays.append(tokenized_chunks)
        count += 1
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


def chunks(sentences1):
    i = 0
    sentences = [x.strip() for x in sentences1 if x]  #to remove empty in sequence
    while len(sentences) < no_of_chunks:
        sentences.append(sentences[i])
        i += 1
    sentences = np.array(sentences)
    return_arr = np.array_split(sentences, no_of_chunks)
    return_arr = [x for x in return_arr if x.size > 0]
    return return_arr


def avg_chunk_word_encoding(essays,embedding_matrix):
    essays_list = []
    count = 0
    for essay in essays:
        # print(count)
        chunks_list = []
        for chunk in essay:
            # print(chunk)
            temp_word_matrix = []
            for word_idx in chunk:
                temp_word_matrix.append(embedding_matrix[word_idx])
            # print("word matrix", np.shape(temp_word_matrix), len(chunk))
            avg = np.mean(temp_word_matrix,axis=0)
            #flatten
            chunks_list.append(avg[0]) # 1 x 200
        count += 1
        essays_list.append(np.array(chunks_list))
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