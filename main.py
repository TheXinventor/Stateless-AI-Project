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

            self.data.append(np.ones((nodeCount, edgeCount)))
    
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

    def forwardPass(self, input: list[float]):
        assert len(input) == self.getNodeCount(0)
        nodes = input
        for i in range(0, self.getLayerCount()):
            nodes = np.matmul(nodes, self.getLayer(i))
            nodes = 1.0 / (1.0 + np.exp(-nodes))
        return nodes

    def forwardPassDetailed(self, input: list[float]):
        assert len(input) == self.getNodeCount(0)
        layers = [input]
        for i in range(0, self.getLayerCount()):
            layers.append(np.matmul(layers[-1], self.getLayer(i)))
            layers[-1] = 1.0 / (1.0 + np.exp(-layers[-1]))
        return layers

    
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

    def __init__(self, input, output, layerSizes, lr = 1.0, maxError = 0.2, maxEpoch = 100):

        assert len(input) == layerSizes[0]
        assert len(output) == layerSizes[-1]

        self.tensor = Tensor(layerSizes)
        epoch = 0

        while np.sum(np.abs(output - self.tensor.forwardPass(input))) > maxError and epoch < maxEpoch:

            # Forward pass
            actual = self.tensor.forwardPassDetailed(input)

            # Backpropogate nodes
            error = [output - actual[-1]]
            for i in range(self.tensor.getLayerCount() - 1, 0, -1):
                error.append(
                    actual[i] * (1.0 - actual[i]) * np.matmul(
                        error[-1],
                        np.rot90(
                            np.fliplr(
                                self.tensor.getLayer(i)
                            )
                        )
                    )
                )
            error.reverse()

            # Update weights
            for i in range(0, self.tensor.getLayerCount()):
                for j in range(0, self.tensor.getNodeCount(i)):
                    for k in range(0, self.tensor.getNodeCount(i+1)):
                        self.tensor.setEdge(i, j, k, lr * actual[i][j] * error[i][k])

            epoch += 1

class Model:

    # Object functions

    def __init__(self, layerSizes: list[int], lr = 1.0, maxError = 0.2, maxEpoch = 100):
        self.lr = lr
        self.maxError = maxError
        self.maxEpoch = maxEpoch

        self.trainCount = np.uint64()
        self.tensor = Tensor(layerSizes)

        #self.iotype = np.dtype([("input", np.float32, layerSizes[0]), ("output", np.float32, layerSizes[-1])])

    def train(self, input, output):
        mask = Mask(
            input,
            output,
            self.tensor.getNodeCounts(),
            lr = self.lr,
            maxError = self.maxError,
            maxEpoch = self.maxEpoch
        )
        self.tensor = (self.tensor * self.trainCount + mask.tensor) / (self.trainCount + 1)
        self.trainCount += 1

    def untrain(self, input, output):
        pass

    def query(self, input):
        return self.tensor.forwardPass(input)

    def trainAll(self, ioList: list[tuple[list, list]]):
        self.tensor *= self.trainCount
        for i in range(0, len(ioList)):
            print("Training example " + str(i) + "/" + str(len(ioList)))
            io = ioList[i]
            mask = Mask(io[0], io[1], self.tensor.getNodeCounts())
            self.tensor += mask.tensor
        self.trainCount += len(ioList)
        self.tensor = self.tensor / self.trainCount
    

