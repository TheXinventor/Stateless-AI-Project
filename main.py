import numpy as np

class Model:

    # Class functions

    def getMask(input, output, layerSizes: list[int]):

        mask = [np.ndarray([layerSizes[i], layerSizes[i+1]]) for i in range(0, len(layerSizes) - 1)]
        
        # Forward propogation
        initial = [input]
        for i in range(0, len(layerSizes) - 1):
            initial.append([np.sum(initial[i]) * np.pow(0.5, layerSizes[i+1])] * layerSizes[i+1]) # Simulates forward pass with all weights as 0.5

        actual = output

        # Back propogation
        for i in range(len(layerSizes) - 2, -1, -1):
            mask[i] = np.ndarray(
                [
                    [
                        Model.getMaskCell(inputVal, outputVal)
                    ] for inputVal in initial[i]
                ] for outputVal in actual
            )
            actual = [np.sum([mask[i][row][col] for row in range(0, len(actual))]) for col in range(0, len(initial[i]))]

        return mask

    def getMaskLayer(input, output):
        inputSize = np.size(input)
        outputSize = np.size(output)

        maskLayer = np.ndarray((inputSize, outputSize))
        for x in range(0, inputSize):
            for y in range(0, outputSize):
                maskLayer[x][y] = Model.getMaskCell(input[x], output[y])
        return maskLayer
                
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
        mask = Model.getMask(input, output, self.layerSizes)
        for i in range(0, len(self.layerSizes) - 1):
            self.layerEdges[i] = (self.layerEdges[i] * self.trainCount + mask[i]) / (self.trainCount + 1)
        self.trainCount += 1

    def untrain(self, input, output):
        mask = Model.getMask(input, output, self.layerSizes)
        for i in range(0, len(self.layerSizes) - 1):
            self.layerEdges[i] = (self.layerEdges[i] * self.trainCount - mask[i]) / (self.trainCount - 1)
        self.trainCount -= 1

    def query(self, input):
        nodes = input
        for layer in self.layerEdges:
            nodes = np.matmul(nodes, layer)
        return nodes