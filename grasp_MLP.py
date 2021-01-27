# pytorch mlp for regression
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 80)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(80, 40)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

        self.hidden3 = Linear(40, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

        self.hidden4 = Linear(10, 8)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()

        # third hidden layer and output
        self.lastlayer = Linear(8, 1)
        xavier_uniform_(self.lastlayer.weight)


    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        # third hidden layer and output
        X = self.lastlayer(X)
        return X

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.001)
    # enumerate epochs
    for epoch in range(200):
        # enumerate mini batches
        ls=.0
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(torch.squeeze(yhat), targets)
            ls=ls+loss.item()
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            if i==19:
                print(f'epoch: {epoch:3} loss: {ls:10.8f}')
                plt.cla()
                ax.plot(targets.data.numpy(), color="orange")
                ax.plot(yhat.data.numpy(), 'g-', lw=3)
                plt.show()
                plt.pause(0.2)  # Note this correction

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    plt.cla()
    ax.plot(actuals, color="orange")
    ax.plot(predictions, 'g-', lw=3)
    ax.set_title('Testing set', fontsize=35)
    plt.show()
    plt.pause(5)
    return mse

###########
datafile1='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TrainData.mat'
datafile2='/Users/long/Documents/BCI/matlab_scripts/force/pls/move4TestData.mat'
raw1=scipy.io.loadmat(datafile1)
raw2=scipy.io.loadmat(datafile2)
train=raw1['train'] # (6299, 115)
test=raw2['test'] #(2699, 115)
tmp=np.concatenate((train,test),0) # (8998, 115)
x=torch.FloatTensor(tmp[0:8400,0:-1]) #torch.Size([8998, 114])
y=torch.FloatTensor(tmp[0:8400,-1]) #torch.Size([8998])
x1=torch.FloatTensor(tmp[8400:,0:-1]) #torch.Size([8998, 114])
y1=torch.FloatTensor(tmp[8400:,-1]) #torch.Size([8998])

class dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)
#train_data = trainData(torch.FloatTensor(trainx),torch.FloatTensor(trainy))
train_data = dataset(x,y)
test_data = dataset(x1,y1)
train_loader = DataLoader(dataset=train_data, batch_size=299, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=299, shuffle=False)
##################

# define the network
input_size=114
model = MLP(input_size)
fig, ax = plt.subplots(figsize=(12,7))
plt.ion()
# train the model
train_model(train_loader, model)
# evaluate the model
mse = evaluate_model(test_loader, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))