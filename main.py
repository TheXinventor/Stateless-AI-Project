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
            return (np.exp((inputVal - output[nodeI]) ** 2) * 2.0 * np.pi) ** -0.5

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

    # Object functions

    def __init__(self, layerSizes: list[int], variance = 1.0):
        self.variance = variance
        self.trainCount = np.uint64()

        self.tensor = Tensor(layerSizes)

    def train(self, input, output):
        mask = Mask(input, output, self.tensor.getNodeCounts())
        self.tensor = (self.tensor * self.trainCount + mask.tensor) / (self.trainCount + 1)
        self.trainCount += 1

    def untrain(self, input, output):
        pass

    def query(self, input):
        nodes = input
        for i in range(0, self.tensor.getLayerCount()):
            nodes = np.matmul(nodes, self.tensor.getLayer(i))
        return nodes

    def trainAll(self, ioList: list[tuple[list, list]]):
        self.tensor *= self.trainCount
        for i in range(0, len(ioList)):
            print("Training example " + str(i) + "/" + str(len(ioList)))
            io = ioList[i]
            mask = Mask(io[0], io[1], self.tensor.getNodeCounts())
            self.tensor += mask.tensor
        self.trainCount += len(ioList)
        self.tensor = self.tensor / self.trainCount
    

