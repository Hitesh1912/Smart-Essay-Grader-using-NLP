import subprocess
import numpy as np
import json

data = open('dict_of_chunked_essays3.txt', 'r').read()
data = eval(data)
data_dic = {}
text = ""
for key in data:
    text += " ".join(data[key])

file_write = open('text8', 'w')
file_write.write(text)
file_write.close()

# script = subprocess.call(['./GloVe-master/demo.sh'], shell=True)
# script.wait()
# vector = open('vectors.txt', 'r').readlines()
# dic = {}
# for line in vector:
#     line_arr = line.split(" ")
#     word = line_arr[0]
#     temp_array = []
#     for i in range(len(line_arr)):
#         if i != 0:
#             temp_array.append(float(line_arr[i]))
#     dic[word] = temp_array
#
# data_dic[key] = dic
# text_arr = text.split(" ")
# x_vals = []
# y_vals = []
# all_vals = dic
# for word in text_arr:
#     word = word.lower()
#     if word not in all_vals:
#         x_vals.append(all_vals['<unk>'])
#         y_vals.append(0)
#     else:
#         x_vals.append(all_vals[word])
#         y_vals.append(1)
# print('bae')
#
# chunk_file = open('chunk_vector.txt', 'w')
# chunk_file.write(str(data_dic))
