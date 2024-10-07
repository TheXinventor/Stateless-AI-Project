import numpy as np

class model():
    matrix: np.ndarray
    maskCount: np.uint32

def getMask(inLayer: np.ndarray, outLayer: np.ndarray) -> np.ndarray:
    edges = np.ndarray((inLayer.shape[0], outLayer.shape[0]))
    for x in range(0, inLayer.shape[0]):
        for y in range(0, outLayer.shape[0]):
            if np.logical_xor(inLayer[x], outLayer[y]):
                edges[x][y] = 0
            else:
                edges[x][y] = 1
    return edges

def addMask(model: model, mask: np.ndarray):
    model.matrix = (model.matrix * model.maskCount + mask) / (model.maskCount + 1)
    model.maskCount += 1

def removeMask(model: model, mask: np.ndarray):
    model.matrix = (model.matrix * model.maskCount - mask) / (model.maskCount - 1)
    model.maskCount -= 1

def getOutLayer(model: model, inLayer: np.ndarray) -> np.ndarray:
    return inLayer.reshape(-1, 1) * model.matrix