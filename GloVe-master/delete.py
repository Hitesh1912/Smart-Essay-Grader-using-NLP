import numpy as np
vector = open('vectors.txt', 'r').readlines()
dic = {}
for line in vector:
    line_arr = line.split(" ")
    word = line_arr[0]
    temp_array = []
    for i in range(len(line_arr)):
        if i != 0:
            temp_array.append(float(line_arr[i]))
    dic[word] = np.array(temp_array)

data_dic = {}
chunk_arr = []
chunk_arr.append(dic)
data_dic[1] = chunk_arr

chunk_file = open('chunk_vector.txt', 'w')
chunk_file.write(str(data_dic))



sentence = "Dear local newspaper I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people helps us learn about the globe(astronomy) and keeps us out of troble Thing about Dont you think so? How would you feel if your teenager is always on the phone with friends Do you ever time to chat with your friends or buisness partner about things"

sentence_arr = sentence.split(" ")
x_vals = []
y_vals = []
all_vals = chunk_arr[0]
print(all_vals)
exit()
for word in sentence_arr:
    word = word.lower()
    if word not in all_vals:
        x_vals.append(all_vals['<unk>'])
        y_vals.append(0)
    else:
        x_vals.append(all_vals[word])
        y_vals.append(1)



from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(np.array(x_vals), np.array(y_vals), epochs=3, validation_split=0.4)


print(data_dic[1])