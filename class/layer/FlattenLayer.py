import numpy as np

class FlattenLayer():
  def __init__(self):
    self.depth = None
    self.weight = None
    self.height = None

  def forward(self, inputData):
    self.height, self.weight, self.depth = inputData.shape
    return np.ravel(inputData)

  def backward(self, dError_dOutput):
    return dError_dOutput.reshape(self.height, self.weight, self.depth)

  def updateWeightBias(self, learningRate): 
    pass

  def getData(self):
    return [
      {
        'type': 'flatten',
        'params': {}
      }
    ]
  
  def getName(self):
    return 'Flatten'
  
  def getShapeOutput(self):
    return (None, self.height * self.weight * self.depth)
  
  def getParameterCount(self):
    return 0

### TESTING ###
if __name__ == "__main__":
  inputData = np.array(
    [
      [
        [1, 2], 
        [3, 4]
      ], 
      [
        [5, 6], 
        [7, 8]
      ]
    ]
  )
  flattenLayer = FlattenLayer()
  newMatrix = flattenLayer.forward(inputData)
  dE_dO = np.array([1, 2, 3, 4, 5, 6, 7, 8])
  originalData = flattenLayer.backward(dE_dO)
  print(inputData == originalData)