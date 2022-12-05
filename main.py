# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NN():

    def __init__(self, i=2, o=2, h=2):
        self.inputNumber = i
        self.outputNumber = o
        self.hiddenNumber = h
        self.momentumRate = 0.01
        self.learningRate = 0.002

        self.weightInputToHidden = np.random.rand(self.inputNumber, self.hiddenNumber)
        self.weightHiddenToOutput = np.random.rand(self.hiddenNumber + 1, self.outputNumber)
        self.deltaWeightInputToHidden = np.zeros((self.inputNumber, self.hiddenNumber + 1))
        self.deltaWeightHiddenToOutput = np.zeros((self.hiddenNumber + 1, self.outputNumber))

        # self.weightInputToHidden = np.array([[0.6,0.8], [0.7,0.8]])
        # self.weightHiddenToOutput = np.array([[0.5,0.7],[0.5,0.9],[0.4,0.5]])

    def sigmoid(self, s):
        lamb = 0.7
        return 1.0 / (1.0 + np.exp(-s * lamb))

    def sigmoidDerivative(self, s):
        return s * (1.0 - s)

    def forward(self, x):
        v = np.matmul(x, self.weightInputToHidden)
        # print(v)
        self.h = self.sigmoid(v)
        self.h1 = np.insert(self.h, len(self.h[0]), 1, axis=1)
        y = np.matmul(self.h1, self.weightHiddenToOutput)
        return self.sigmoid(y)

    def backward(self, x, y, predictedY):
        self.o_error = y - predictedY
        # print(o_error)
        o_delta = self.o_error * self.sigmoidDerivative(predictedY)
        h_error = np.matmul(o_delta,
                            np.matrix.transpose(self.weightHiddenToOutput)) * self.learningRate
        h_delta = h_error * self.sigmoidDerivative(self.h1)

        self.deltaWeightInputToHidden *= self.momentumRate
        self.deltaWeightHiddenToOutput *= self.momentumRate
        self.deltaWeightInputToHidden += np.matmul(np.matrix.transpose(x),
                                                   h_delta) * self.learningRate
        self.deltaWeightHiddenToOutput += np.matmul(np.matrix.transpose(self.h1),
                                                    o_delta) * self.learningRate
        self.weightInputToHidden += self.deltaWeightInputToHidden[:, :len(self.deltaWeightInputToHidden[0]) - 1]
        self.weightHiddenToOutput += self.deltaWeightHiddenToOutput

    def predict(self, x):
        return self.forward(x)


##nn = NN()
##x = np.array([[1,1],[0.5,0.4],[0.1,1],[1,0.7]])
##y = np.array([[0.5,0.5],[0.25,0.2],[0.05,0.5],[0.5,0.35]])
##for i in range( 10000):
##    #print('epoch ', i)
##    px = nn.forward(x)
##    nn.backward(x,y,px)
##
##px = np.array([[0.3,0.2],[0.15,0.24]])
##print(nn.predict(x))
##print(nn.predict(px))


import csv

nn = NN()


n = 3000

def processData(data):
    x = np.array([])
    y = np.array([])

    for row in data:
        if len(row) != 4:
            print('error found')
            continue

        x = np.concatenate((x, [row[0], row[1]]))
        y = np.concatenate((y, [row[2], row[3]]))

    x = x.reshape(len(x) // 2, 2)
    y = y.reshape(len(y) // 2, 2)
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    scaler.fit(y)
    y = scaler.transform(y)
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')

    return (x,y)

with open('normalised.csv', 'r') as file:
    reader = csv.reader(file)

    first_row = next(reader, None)
    data = list(reader)

    training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)
    print(len(training_data[0]))
    print(len(testing_data[0]))
    x, y = processData(training_data)

error_list = []

for i in range(20000):
   px = nn.forward(x)
   nn.backward(x,y,px)
   error_list.append(np.square(nn.o_error).mean())

for i in range(0,n,20):
   print(i)
   print(x[i])
   print(y[i])
   print(nn.predict([x[i]]))


plt.plot(range(len(error_list)), error_list)
plt.title('Mean Sum Squared Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# import torch
# import torch.nn as nn
#
# model = nn.Sequential(nn.Linear(2, 2),
#                       nn.Sigmoid(),
#                       nn.Linear(2, 2),
#                       nn.Sigmoid())
# print(model)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# losses = []
# for epoch in range(30000):
#     pred_y = model(torch.from_numpy(x).to(torch.float32))
#     loss = loss_function(pred_y, torch.from_numpy(y).to(torch.float32))
#     losses.append(loss.item())
#
#     model.zero_grad()
#     loss.backward()
#
#     optimizer.step()
#
#
#
#
# for i in range(0,n,20):
#    print(i)
#    print(pred_y[i])
#    print(y[i])
#
# import matplotlib.pyplot as plt
#
# plt.plot(losses)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

