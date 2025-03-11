import numpy as np

class Tensor:

    data: list[np.ndarray]

    # Utilities
    def __init__(self, layerSizes: list[int]):

        assert len(layerSizes) >= 2

        # Initialize edges to 0
        self.data = []
        for i in range(0, len(layerSizes) - 1):
            nodeCount = layerSizes[i]
            edgeCount = layerSizes[i+1]

            self.data.append(np.zeros((nodeCount, edgeCount)))
    
    def __str__(self):
        outStr = ""
        for layer in self.data:
            for node in layer:
                outStr += str(node) + "\n"
            outStr += "\n"
        return outStr
    
    def __add__(self, other):
        output = Tensor(self.getNodeCounts())
        for i in range(0, self.getLayerCount()):
            output.setLayer(i, self.getLayer(i) + other.getLayer(i))
        return output

    def __mul__(self, other: int):
        output = Tensor(self.getNodeCounts())
        for i in range(0, self.getLayerCount()):
            output.setLayer(i, self.getLayer(i) * other)
        return output

    def __truediv__(self, other: float):
        output = Tensor(self.getNodeCounts())
        for i in range(0, self.getLayerCount()):
            output.setLayer(i, self.getLayer(i) / other)
        return output
    
    # Getters
    def getNodeCounts(self):
        return [self.getNodeCount(i) for i in range(0, self.getLayerCount() + 1)]

    def getLayerCount(self):
        return len(self.data)
    
    def getNodeCount(self, layerI):
        if layerI < len(self.data):
            return len(self.data[layerI])
        else:
            return len(self.data[-1][0])

    def getLayer(self, layerI):
        return self.data[layerI]
    
    def getNode(self, layerI, nodeI):
        return self.data[layerI][nodeI]
    
    def getEdge(self, layerI, nodeI, edgeI):
        return self.data[layerI][nodeI][edgeI]
    
    # Setters
    def setLayer(self, layerI, value):
        self.data[layerI] = value
    
    def setNode(self, layerI, nodeI, value):
        self.data[layerI][nodeI] = value
    
    def setEdge(self, layerI, nodeI, edgeI, value):
        self.data[layerI][nodeI][edgeI] = value

class Mask:

    tensor: Tensor

    def __init__(self, input, output, layerSizes):

        assert len(input) == layerSizes[0]
        assert len(output) == layerSizes[-1]

        self.tensor = Tensor(layerSizes)

        # Add up all routes
        for i in range(0, len(input)):
            self.initRecursive(0, i, input[i], output)

        # Divide each edge by no. of routes using that edge
        product = np.prod(layerSizes)
        for i in range(0, len(layerSizes) - 1):
            self.tensor.setLayer(i, self.tensor.getLayer(i) * layerSizes[i] * layerSizes[i+1] / product)
    
    def initRecursive(self, layerI, nodeI, inputVal, output):

        # If on final layer, compare input and output
        if layerI == self.tensor.getLayerCount():
            return Model.gaussianDist(inputVal, output[nodeI], 1.0)
        
        # Loop through edges from this node
        nodeVal = 0.0

        for edgeI in range(0, self.tensor.getNodeCount(layerI+1)):

            # Get value from next iteration
            edgeVal = self.initRecursive(layerI+1, edgeI, inputVal, output)

            # Add edge value to mask
            edgeTotal = self.tensor.getEdge(layerI, nodeI, edgeI) + edgeVal
            self.tensor.setEdge(layerI, nodeI, edgeI, edgeTotal)

            # Add edge to total
            nodeVal += edgeVal
        
        # Return sum of outgoing edges
        return nodeVal

class Model:

    '''
    # Class functions

    def getMask(input, output, layerSizes: list[int]):

        mask = [np.ndarray([layerSizes[i], layerSizes[i+1]]) for i in range(0, len(layerSizes) - 1)]

        # Loop through every path through the network

        # Get normal distribution of difference between input and output

        # 

        return mask
    
    def getMaskRecursive(nodeIndex, layerIndex, layerSizes, inputVal, output):

        # If on final layer, compare input and output
        if layerIndex == len(layerSizes) - 1:
            return Model.gaussianDist(inputVal, output[nodeIndex], 1.0)
        
        # Loop through edges from this node
        for edgeIndex in range(0, len(layerSizes[layerIndex+1])):

            # Get value from next iteration
            edgeVal = getMaskRecursive(edgeIndex, layerIndex+1, layerSizes, inputVal, output)

            # Add edge value to mask
            output[layerIndex][nodeIndex][edgeIndex] += edgeVal

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
    
    '''
    
    # Math functions
    
    def gaussianDist(x, mean, variance):
        return np.power(np.e, -np.square(x - mean) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
    
    def linear(x):
        return x

    # Object functions

    def __init__(self, layerSizes: list[int], variance = 1.0):
        self.variance = variance
        self.trainCount = np.uint64()

        self.tensor = Tensor(layerSizes)

    def train(self, input, output):
        mask = Mask(input, output, self.tensor.getNodeCounts())
        self.tensor = (self.tensor * self.trainCount + mask.tensor) / (self.trainCount + 1)
        self.trainCount += 1

    def query(self, input):
        nodes = input
        for i in range(0, self.tensor.getLayerCount()):
            nodes = np.matmul(nodes, self.tensor.getLayer(i))
        return nodes
    
    '''
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
    
    def getMaskRecursive(self, nodeIndex, layerIndex, inputVal, output):

        # If on final layer, compare input and output
        if layerIndex == len(self.layerSizes) - 1:
            return Model.gaussianDist(inputVal, output[nodeIndex], 1.0)
        
        # Loop through edges from this node
        for edgeIndex in range(0, len(layerSizes[layerIndex+1])):

            # Get value from next iteration
            edgeVal = self.getMaskRecursive(edgeIndex, layerIndex+1, inputVal, output)

            # Add edge value to mask
            output[layerIndex][nodeIndex][edgeIndex] += edgeVal
    '''



    

