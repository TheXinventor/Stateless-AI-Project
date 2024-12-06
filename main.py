import numpy as np

class Model:

    # Class functions

    def getMask(input, output):
        inputSize = np.size(input)
        outputSize = np.size(output)

        mask = np.ndarray((inputSize, outputSize))
        for x in range(0, inputSize):
            for y in range(0, outputSize):
                mask[x][y] = Model.getMaskCell(input[x], output[y])
        return mask
                
    def getMaskCell(inputVal, outputVal):
        # Calculates standard normal distribution of difference between values
        return np.pow(np.e, -np.square(inputVal - outputVal) / 2) / np.sqrt(2 * np.pi)
    
    # Math functions
    
    def gaussianDist(x, mean, variance):
        return np.pow(np.e, -np.square(x - mean) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    
    def linear(x):
        return x

    # Object functions

    def __init__(self, layerSizes: list[int], activationFunc = Model.linear, variance = 1.0):
        
        assert len(layerSizes) >= 2

        self.layerSizes = layerSizes
        self.activationFunc = activationFunc
        self.variance = variance

        self.layerEdges = [np.ndarray([layerSizes[i], layerSizes[i+1]]) for i in range(0, len(layerSizes) - 1)]
        self.trainCount = np.uint64()

    def train(self, input, output):
        pass

    def untrain(self, input, output):
        pass

    def query(self, input):
        pass