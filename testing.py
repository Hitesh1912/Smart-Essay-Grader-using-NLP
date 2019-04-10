
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utilities import  *
import time
from itertools import zip_longest
from nltk import ngrams



vocabulary_size = 5000


def word_tokenize(essay_data, essay_list):
    tokenizer = Tokenizer(num_words=vocabulary_size)
    print('fitting tokenizer on whole essay')
    tokenizer.fit_on_texts(essay_data)
    print('fitting complete')
    tokenized_essays = []
    temp_list = []
    # tokenized_chunks = []
    # this takes our sentences and replaces each word with an integer
    for essay in essay_list:
        tokenized_chunks = []
        for chunk in essay:
            chunk_seq = tokenizer.texts_to_sequences(chunk)
            # print(chunk_seq)
            print(tokenizer.sequences_to_texts(chunk_seq))
            # we then pad the sequences so they're all the same length (sequence_length)
            # chunk = pad_sequences(chunk, 50, padding='post')  #check
            tokenized_chunks.append(chunk_seq)
        tokenized_essays.append(tokenized_chunks)


    return tokenized_essays, tokenizer



essay_data = "dear local newspaper think effects computers people great learning skills affects give time chat friends new people helps learn globe astronomy keeps troble  thing about  dont think so would feel teenager always phone friends  ever time chat friends buisness partner things well there new way chat computer plenty sites internet so  organization1 organization2 caps1 facebook myspace ect think setting meeting boss computer teenager fun phone rushing get cause want use it learn countrys states outside yours well computer internet new way learn going time  might think child spends lot time computer ask question economy sea floor spreading even date1 surprise much he she knows believe computer much interesting class day reading books child home computer local library better friends fresh perpressured something know isnt right might know child is caps2 forbidde hospital bed drive - by rather child computer learning chatting playing games safe sound home community place hope reached point understand agree me computers great effects child gives time chat friends new people helps learn globe believe keeps troble thank listening"
essay_data = essay_data.split(" ")
essays = [['dear local newspaper think effects computers people great learning skills affects give time chat friends new people helps learn globe astronomy keeps troble  thing about  dont think so would feel teenager always phone friends  ever time chat friends buisness partner things', 'well there new way chat computer plenty sites internet so  organization organization caps facebook myspace ect', 'think setting meeting boss computer teenager fun phone rushing get cause want use', 'learn countrys states outside yours well computer internet new way learn going time  might think child spends lot time computer ask question economy sea floor spreading even date surprise much he she knows', 'believe computer much interesting class day reading books', 'child home computer local library better friends fresh perpressured something know isnt right', 'might know child is caps forbidde hospital bed drive - by', 'rather child computer learning chatting playing games safe sound home community place', 'hope reached point understand agree me computers great effects child gives time chat friends new people helps learn globe believe keeps troble', 'thank listening']]

essay_list = []
for essay in essays:
    data_arr = []
    for chunk in essay:
        chunk = chunk.split(" ")
        data_arr.append(chunk)
    essay_list.append(data_arr)

# print(essay_arr[0][0][0])
# exit()

tokens, tokenizer = word_tokenize(essay_data,essay_list)
print(tokens)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))