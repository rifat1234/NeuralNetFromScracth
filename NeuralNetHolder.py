import numpy as np

class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.lambdaSig = 0.8

        self.weightInputToHidden = np.array([[-3.13001323, -0.84761436,  -0.96318417, 0.93596882,
                                              -1.15475506, 0.82745629, -0.43764451],
                                    [-1.78095091, -0.64834933, -4.50052592,  -0.95868441,
                                     -1.13516041,  -0.83447537, -2.853553]],
                                            dtype='float64')

        #one extra row of weight is for bias
        self.weightHiddenToOutput = np.array(
                                    [[ -24.88127729,  -1.13430235],
                                    [ -2.35247789, -1.69568349],
                                     [ 23.14276452,  -0.91228577],
                                     [-14.00291846,   2.80799713],
                                     [  3.07591799, -10.53786565],
                                     [-12.49497526,   3.12259251],
                                     [ 10.92020608,   4.5919809 ],
                                     [ 11.72383839,  -0.33708746]],  dtype='float64')

    def sigmoid(self, s):
        lamb = self.lambdaSig
        return 1.0 / (1.0 + np.exp(-s * lamb))
    
    def forward(self, x):
        v = np.matmul(x, self.weightInputToHidden)
        self.h = self.sigmoid(v)
        self.h1 = np.insert(self.h, len(self.h[0]), 1, axis=1) # 1 added as bias for the hidden layer
        y = np.matmul(self.h1, self.weightHiddenToOutput)
        return self.sigmoid(y)

    def denormalize(self, val, mn, mx):
        return (val * (mx - mn)) + mn
    
    def predict(self, input_row):
        y0min = -7.600000
        y0max = 8.000000
        y1min = -6.180152
        y1max = 7.213645
        
        x = np.array(input_row.split(","), dtype = 'float64')
        x = x.reshape(len(x) // 2, 2)
        y = self.forward(x)
        y0 = self.denormalize(y[0][0], y0min, y0max )
        y1 = self.denormalize(y[0][1], y1min, y1max )
        
        return y1,y0
