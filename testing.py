
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



vocabulary_size = 5000


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




