import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu, sigmoid

class DenseLayer():
  def __init__(self, numUnit, activationFunctionName):
    self.numUnit = numUnit
    self.activationFunctionName = activationFunctionName
    self.weight = None
    self.bias = np.zeros((numUnit,))

  def forward(self, inputData):
    numFeatures = np.prod(inputData.shape)
    if (self.weight is None) :
      self.weight = np.random.randn(numFeatures, self.numUnit)
    output = np.dot(inputData, self.weight) + self.bias
    if (self.activationFunctionName.lower() == 'relu'):
      output = relu(output)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      output = sigmoid(output)
    return output
  
### TESTING ###
if __name__ == "__main__":
  matrix = np.array(
    [
      [
        [
          [1,11,2],
          [1,10,4],
          [6,12,8],
        ],
        [
          [7,1,2],
          [5,-1,2],
          [7,-4,2],
        ],
        [
          [-2,23,2],
          [2,20,4],
          [8,6,6],
        ]
      ]
    ]
  )
  print(matrix[0].shape)
  print("=====")
  matrix = np.ravel(matrix[0])
  print(matrix.shape)
  denseLayer = DenseLayer(numUnit = 16, activationFunctionName = 'sigmoid')
  newMatrix = denseLayer.forward(matrix)
  print(newMatrix.shape)