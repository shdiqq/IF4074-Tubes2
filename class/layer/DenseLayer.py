import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu, sigmoid, linear

class DenseLayer():
  def __init__(self, numUnit, activationFunctionName):
    self.numUnit = numUnit
    self.activationFunctionName = activationFunctionName
    self.weight = None
    self.deltaWeight = None
    self.bias = np.zeros((numUnit,))
    self.deltaBias = None
    self.output = None

  def forward(self, inputData):
    numFeatures = np.prod(inputData.shape)
    if (self.weight is None) :
      self.weight = np.random.randn(numFeatures, self.numUnit)
    if (self.deltaWeight is None) :
      self.deltaWeight = np.zeros(self.weight.shape)
    output = np.dot(inputData, self.weight) + self.bias
    if (self.activationFunctionName.lower() == 'relu'):
      output = relu(output)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      output = sigmoid(output)
    elif (self.activationFunctionName.lower() == 'linear'):
      output = linear(output)
    # elif (self.activationFunctionName.lower() == 'softmax'):
    #   output = softmax(output)
    self.output = output
    return output
  
  def backward(self, inputData, prevLayerDelta):
    if (self.activationFunctionName.lower() == 'relu'):
      dOutput_dInput = relu(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      dOutput_dInput = sigmoid(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'linear'):
      dOutput_dInput = linear(self.output, derivative=True)
    # elif (self.activationFunctionName.lower() == 'softmax'):
    #   derivativeValues = softmax(self.output, derivative=True)
    dError_dInput = prevLayerDelta * dOutput_dInput
    self.deltaWeight = np.outer(inputData, dError_dInput)
    dError_dInput = np.dot(dError_dInput, self.weight.T)

    # Menghitung gradien bias
    self.deltaBias = dError_dInput

    return dError_dInput

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