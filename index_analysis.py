from itertools import zip_longest
import numpy as np



def grouper(arr, fillvalue=None):
    args = [iter(arr)] * 3
    return zip_longest(fillvalue=fillvalue, *args)



arr = list(range(1,7789))
iter = grouper(arr)

group_list = []
for i in iter:
    # i = " ".join(i).rstrip()
    print(i)
    group_list.append(i)


np.savetxt("prediction_output/group_index.txt",group_list,fmt="%i")

