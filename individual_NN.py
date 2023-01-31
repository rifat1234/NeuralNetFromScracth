import numpy as np
import csv


fileName = 'ce889_dataCollection.csv'
epoch = 90000

class NN():

    def __init__(self, i=2, o=2, h=7, lRate = 0.005, mRate = 0.02, lambSig = 0.8):
        self.inputNumber = i
        self.outputNumber = o
        self.hiddenNumber = h
        self.momentumRate = mRate
        self.learningRate = lRate
        self.lambdaSig = lambSig

        self.weightInputToHidden = np.random.rand(self.inputNumber, self.hiddenNumber)
        self.weightHiddenToOutput = np.random.rand(self.hiddenNumber + 1, self.outputNumber)
        self.deltaWeightInputToHidden = np.zeros((self.inputNumber, self.hiddenNumber + 1)) # hidden number increased by one because of bias
        self.deltaWeightHiddenToOutput = np.zeros((self.hiddenNumber + 1, self.outputNumber))

    def sigmoid(self, s):
        lamb = self.lambdaSig
        return 1.0 / (1.0 + np.exp(-s * lamb))

    def sigmoidDerivative(self, s):
        return s * (1.0 - s)

    def forward(self, x):
        v = np.matmul(x, self.weightInputToHidden)
        self.h = self.sigmoid(v)
        self.h1 = np.insert(self.h, len(self.h[0]), 1, axis=1) # 1 added as bias for the hidden layer
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
            trainingErrorList.append(np.square(y - px).mean()) # RMSE calculation

            pty = nn.forward(tx)
            testErrorList.append(np.square(ty - pty).mean()) # RMSE calculation

        return (trainingErrorList, testErrorList)


"""
Data Partition   
"""
def splitData(data):
    mask = np.random.rand(len(data)) <= 0.8
    trainingData = data[mask]
    testingData = data[~mask]
    return (trainingData, testingData)


"""
Normalize Data   
"""
def normalize(data):
    x0Min = data[:, 0].min()
    x0Max = data[:, 0].max()
    x1Min = data[:, 1].min()
    x1Max = data[:, 1].max()

    global y0Min,y0Max,y1Min,y1Max

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


"""
Denormalize Data   
"""
def denormalize(val, mn, mx):
    return (val * (mx - mn)) + mn


"""
Processing Data   
"""
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



"""
Reading data from CSV   
"""
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
testMin, epochTest = min((val, idx) for (idx, val) in enumerate(testingErrorList))
trainMin, epochTrain = min((val, idx) for (idx, val) in enumerate(trainingErrorList))
print("After %d epoch, best training accuracy: %lf at %d epochs  & best testing accuracy: %lf at %d epochs"
      % (epoch, trainMin, epochTrain, testMin, epochTest))

print("\n\n\nValues needed for denormalization and neural holder normalization ")
print("y0 min %lf, y0 max %lf \ny1 min %lf, y1 max %lf" %(y0Min, y0Max, y1Min, y1Max))
print("Weight Input to Hidden \n%s" %(nn.weightInputToHidden))
print("Weight Hidden to Input \n%s" %(nn.weightHiddenToOutput))


"""
Matplotlib is used for testing purpose  
"""
import matplotlib.pyplot as plt
plt.plot(range(len(trainingErrorList)), testingErrorList)
plt.title('Mean Sum Squared Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()




"""
Optimisation Code, check for minimum value with various values of hidden layer neurons(h), learning rate(l), momentum rate(m)   
"""
# lRate = [x * 0.001 for x in range(1, 6)]
# mRate = [x * 0.005 for x in range(1, 7)]
# lambSig = [x * 0.1 for x in range(0, 10)]
# print(lRate)
# print(mRate)
# print(lambSig)
#
# mn = 2000.0
# for h in range(2,11):
#     for l in lRate:
#         for m in mRate:
#             nn = NN(2,2,h,l,m,0.7)
#             trainingErrorList, testingErrorList = nn.train(x, tx, epoch)
#             val, idx = min((val, idx) for (idx, val) in enumerate(testingErrorList))
#             if val < mn:
#                 mn = val
#                 print("h = %d, l = %lf, m = %lf, lSig = %lf, min = %lf index = %d" %(h,l,m,0.7, mn, idx))



