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
import csv
import matplotlib.pyplot as plt

fileName = 'ce889_dataCollection.csv'
epoch = 20000
class NN():

    def __init__(self, i=2, o=2, h=3, lRate = 0.003, mRate = 0.015):
        self.inputNumber = i
        self.outputNumber = o
        self.hiddenNumber = h
        self.momentumRate = mRate
        self.learningRate = lRate

        self.weightInputToHidden = np.random.rand(self.inputNumber, self.hiddenNumber)
        self.weightHiddenToOutput = np.random.rand(self.hiddenNumber + 1, self.outputNumber)
        self.deltaWeightInputToHidden = np.zeros((self.inputNumber, self.hiddenNumber + 1))
        self.deltaWeightHiddenToOutput = np.zeros((self.hiddenNumber + 1, self.outputNumber))

    def sigmoid(self, s):
        lamb = 0.7
        return 1.0 / (1.0 + np.exp(-s * lamb))

    def sigmoidDerivative(self, s):
        return s * (1.0 - s)

    def forward(self, x):
        v = np.matmul(x, self.weightInputToHidden)
        self.h = self.sigmoid(v)
        self.h1 = np.insert(self.h, len(self.h[0]), 1, axis=1)
        y = np.matmul(self.h1, self.weightHiddenToOutput)
        return self.sigmoid(y)

    def backward(self, x, y, predictedY):
        self.yError = y - predictedY
        yDelta = self.yError * self.sigmoidDerivative(predictedY)
        hError = np.matmul(yDelta,
                            np.matrix.transpose(self.weightHiddenToOutput)) * self.learningRate
        hDelta = hError * self.sigmoidDerivative(self.h1)

        self.deltaWeightInputToHidden *= self.momentumRate
        self.deltaWeightHiddenToOutput *= self.momentumRate
        self.deltaWeightInputToHidden += np.matmul(np.matrix.transpose(x),
                                                   hDelta) * self.learningRate
        self.deltaWeightHiddenToOutput += np.matmul(np.matrix.transpose(self.h1),
                                                    yDelta) * self.learningRate
        self.weightInputToHidden += self.deltaWeightInputToHidden[:, :len(self.deltaWeightInputToHidden[0]) - 1]
        self.weightHiddenToOutput += self.deltaWeightHiddenToOutput

    def predict(self, x):
        return self.forward(x)

    def train(self, x, tx, epoch):
        testErrorList, trainingErrorList = [],[]

        for i in range(epoch):
            px = nn.forward(x)
            nn.backward(x, y, px)
            trainingErrorList.append(np.square(y - px).mean())

            pty = nn.forward(tx)
            testErrorList.append(np.square(ty - pty).mean())

        return (trainingErrorList, testErrorList)


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

def splitData(data):
    mask = np.random.rand(len(data)) <= 0.8
    trainingData = data[mask]
    testingData = data[~mask]
    return (trainingData, testingData)

def normalize(data):
    x0Min = data[:, 0].min()
    x0Max = data[:, 0].max()
    x1Min = data[:, 1].min()
    x1Max = data[:, 1].max()

    y0Min = data[:, 2].min()
    y0Max = data[:, 2].max()
    y1Min = data[:, 3].min()
    y1Max = data[:, 3].max()

    def norm(val,mn,mx):
        return (val - mn) / (mx - mn)

    for i in range(len(data)):
        data[i][0] = norm(data[i][0],x0Min,x0Max)
        data[i][1] = norm(data[i][1],x1Min,x1Max)
        data[i][2] = norm(data[i][2],y0Min,y0Max)
        data[i][3] = norm(data[i][3],y1Min,y1Max)

    return data

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
    return (x,y)

def readData(fileName) :
    with open(fileName, 'r') as file:
        reader = csv.reader(file)

        first_row = next(reader, None)
        data = list(reader)
        data = np.array(data)
        data = np.asarray(data, dtype='float64')
        return data




data = readData(fileName)
data = normalize(data)
training_data, testing_data = splitData(data)
print("Training Set Size: %d" %len(training_data))
print("Testing Set Size: %d" %len(testing_data))

x, y = processData(training_data)
tx,ty = processData(testing_data)

nn = NN()
trainingErrorList, testingErrorList = nn.train(x, tx, epoch)
print("After %d epoch, training accuracy: %0.4f & testing accuracy: %0.4f" % (epoch, trainingErrorList[len(trainingErrorList)-1], testingErrorList[len(testingErrorList)-1]))


plt.plot(range(len(trainingErrorList)), trainingErrorList)
plt.title('Mean Sum Squared Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



#
# import torch
# import torch.nn as nn
#
# model = nn.Sequential(nn.Linear(2, 3),
#                       nn.Sigmoid(),
#                       nn.Linear(3, 2),
#                       nn.Sigmoid())
# print(model)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.702)
# losses = []
# for epoch in range(20000):
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
# import matplotlib.pyplot as plt
#
# plt.plot(losses)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

