import subprocess
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding


data = open('../dict_of_chunked_essays.txt', 'r').read()
data = eval(data)
data_dic = {}
for key in data:
    file_write = open('text8', 'w')
    text = " ".join(data[key])
    file_write.write(text)
    file_write.close()

    script = subprocess.call(['./demo.sh'], shell=True)
    # script.wait()
    vector = open('vectors.txt', 'r').readlines()
    dic = {}
    for line in vector:
        line_arr = line.split(" ")
        word = line_arr[0]
        temp_array = []
        for i in range(len(line_arr)):
            if i != 0:
                temp_array.append(float(line_arr[i]))
        dic[word] = temp_array

    data_dic[key] = dic
    text_arr = text.split(" ")
    x_vals = []
    y_vals = []
    all_vals = dic
    for word in text_arr:
        word = word.lower()
        if word not in all_vals:
            x_vals.append(all_vals['<unk>'])
            y_vals.append(0)
        else:
            x_vals.append(all_vals[word])
            y_vals.append(1)

    model = Sequential()
    model.add(Embedding(20000, 100, input_length=50))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(x_vals), np.array(y_vals), epochs=3, validation_split=0.4)
    print('bae')

chunk_file = open('chunk_vector.txt', 'w')
chunk_file.write(str(data_dic))
