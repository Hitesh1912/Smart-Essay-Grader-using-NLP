# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
# from keras.layers.embeddings import Embedding
#
# ## Plotly
# import plotly.offline as py
# import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)
# # Others
# import nltk
# import string
# import numpy as np
# import pandas as pd
# from nltk.corpus import stopwords
#
# from sklearn.manifold import TSNE



file_read = open('chunk_vector.txt', 'r').read()
dictionary = eval(file_read)
print("bae")
exit()


# model = Sequential()
# model.add(Embedding(20000, 100, input_length=50))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#
# model.fit()