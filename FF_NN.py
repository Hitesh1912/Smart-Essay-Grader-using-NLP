import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
np.random.seed(7)


# Function importing Dataset
def importdata():
    data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
        sep=',', header=None)
    print(data.shape)
    return data.values


# Function to split the dataset
def splitdataset(data):
    # Seperating the target variable
    X = data[:, 0:data.shape[1]-1]
    Y = data[:, data.shape[1]-1]
    # print(np.shape(X), np.shape(Y))
    return X, Y

def mean_squared_error(actual, predicted):
    mse = (np.square(np.array(actual) - np.array(predicted))).mean()
    return mse

# Calculate accuracy
def accuracy_val(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def run_model(X,y,X_test,y_test):
    # batch_size = 50
    #evaluate weights after every 30 batch
    # create model
    model = Sequential()
    # model.add(Dense(200, input_dim=8))

    #ADD THE LSTM HIDDEN LAYER AS INPUT
    model.add(Dense(10,input_dim=57, activation='relu')) #hidden layer
    model.add(Dense(1, activation='sigmoid')) #output layer

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # learning rate

    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    # Fit the model
    model.fit(X, y, epochs=100, batch_size=50)
    print("training complete...")

    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    # print("MSE",mean_squared_error(y,rounded))

    print("train accuracy",accuracy_val(y,rounded))

    # calculate predictions
    predictions_t = model.predict(X_test)
    # round predictions
    rounded_t = [round(x[0]) for x in predictions_t]
    print(rounded_t)
    # print("MSE",mean_squared_error(y,rounded))
    print("test accuracy", accuracy_val(y_test, rounded_t))








if __name__ == '__main__':
    data_set = importdata()
    X, y = splitdataset(data_set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    # print(np.shape(X_train))
    run_model(X_train,y_train,X_test,y_test )
