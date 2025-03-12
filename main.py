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
        return np.power(np.e, -np.square(inputVal - outputVal) / 2) / np.sqrt(2 * np.pi)

    # Object functions

    def __init__(self, inputSize: int, outputSize: int):
        self.inputSize = inputSize
        self.outputSize = outputSize

        self.matrix = np.ndarray((inputSize, outputSize))
        self.trainCount = np.uint64()

    def train(self, input, output):
        self.matrix = (self.matrix * self.trainCount + Model.getMask(input, output)) / (self.trainCount + 1)
        self.trainCount += 1

    def untrain(self, input, output):
        self.matrix = (self.matrix * self.trainCount - Model.getMask(input, output)) / (self.trainCount - 1)
        self.trainCount -= 1
    
    # TODO Fix errors with trainList and untrainList
    def trainList(self, data: list[tuple[list, list]]):
        maskSum = np.sum([Model.getMask(unit[0], unit[1]) for unit in data])
        self.matrix = (self.matrix * self.trainCount + maskSum) / (self.trainCount + len(data))
        self.matrix += len(data)
    
    def untrainList(self, data: list[tuple[list, list]]):
        maskSum = np.sum([Model.getMask(unit[0], unit[1]) for unit in data])
        self.matrix = (self.matrix * self.trainCount + maskSum) / (self.trainCount - len(data))
        self.matrix -= len(data)


    def query(self, input):
        return np.matmul(input, self.matrix).flatten()
