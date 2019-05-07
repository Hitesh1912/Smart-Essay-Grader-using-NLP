# Essay_Grader

![Screen Shot 2019-04-12 at 11 17 41 AM](https://user-images.githubusercontent.com/17843556/57314819-66aa9d80-70c0-11e9-83e5-04b5532e5c5f.png)


We have prepared our data from the ASAP dataset and used Glove vectors representation to get the word vector representation.

We have already processed the data and provided in the folder. kindly include and use processed files mentioned below to be used by the model. 
1. dict_of_chunked_essays.txt
2. list_of_scores.txt
3. vectors.txt


The dataset can be accessed from the link:
https://drive.google.com/drive/folders/1PcQCHDe6i47VKUqAckTG22P3Io9qKyNc?usp=sharing

However the dataset has been preprocessed from the preprocessing.py file and the above mentioned files have been generated as output.


File description
vectors.txt : consist of trained word vectors from GloVe

Preprocessing.py:  performs the cleaning and pre-processing of our dataset and generates the file required by the model

LSTM_v1.py : main file to run the models

helper_functions.py : consist of helper functions used in  LSTM_v1.py

list_of_scores.txt : normalized scores of all the essays

dict_of_chunked_essays.txt: dictionary of all the essays represented as chunks

results.txt: Results of the models


Run the file: LSTM_v1.py
It runs three models:
1. LSTM
2. GRU
3. Vanilla RNN

#Kindly refer the comments mentioned in above file for detailed instructions.

Output:
1. We first get the RMSE and pearson value of our LSTM model and then print out the ids and chunks which are weak
2. After we run our LSTM model, we run GRU and RNN for comparison. We get RMSE and v value for both the model.
