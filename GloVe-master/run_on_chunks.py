import subprocess
import numpy as np
import json

data = open('../dict_of_chunked_essays.txt', 'r').read()
data = eval(data)

data_dic = {}
for key in data:
    chunk_arr = []
    chunk_of_essay = data[key]
    for chunk in chunk_of_essay:
        file_write = open('text8', 'w')
        file_write.write(chunk)
        file_write.close()
        script = subprocess.call(['./demo.sh'], shell=True)
        # script.wait()
        # for line in script.stdout.readlines():
        #     print(line)
        exit()
        vector = open('vectors.txt', 'r').read()
        print(vector)
        exit()
        dic = {}
        print(chunk)
        for line in vector:
            line_arr = line.split(" ")
            word = line_arr[0]
            temp_array = []
            for i in range(len(line_arr)):
                if i != 0:
                    temp_array.append(float(line_arr[i]))
            dic[word] = np.array(temp_array)

        chunk_arr.append(dic)
    data_dic[key] = chunk_arr
    print(data_dic)
    exit()

chunk_file = open('chunk_vector.txt', 'w')
chunk_file.write(str(data_dic))
