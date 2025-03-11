import numpy as np

class Mask:

    matrix: np.ndarray

    def __init__(self, input, output, layerSizes):
        pass

class Model:

    matrix: np.ndarray
    trainCount: int

    def __init__(self, inputSize, outputSize):
        self.matrix = np.zeros((np.power(2, inputSize), outputSize))
        self.trainCount = 0

        print((np.power(2, inputSize), outputSize))

    def train(self, input, output):
        pass

    def query(self, input):
        return np.matmul(input, self.matrix)
    




    

