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
    self.input = None

  def forward(self, inputData):
    self.input = inputData
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
  
  def backward(self, dError_dOutput):
    # print(f"dE/dO = {dError_dOutput}")
    if (self.activationFunctionName.lower() == 'relu'):
      dOutput_dNet = relu(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      dOutput_dNet = sigmoid(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'linear'):
      dOutput_dNet = linear(self.output, derivative=True)
    # elif (self.activationFunctionName.lower() == 'softmax'):
    #   derivativeValues = softmax(self.output, derivative=True)
    # print(f"dO/dNet = {dOutput_dNet}")
    # print(f"dNet/dW = {self.input}")
    deltaError = dError_dOutput * dOutput_dNet
    # print(f"dE/dNet = {deltaError}")
    for i in range (len(self.deltaWeight)):
      self.deltaWeight[i] = self.input[i] * deltaError
    # print(f"dE/dW = {self.deltaWeight}")
    self.deltaBias = deltaError
    # print(f"dE/dBias = {self.deltaBias}")
    dError_dOutputBeforeLayer = np.dot(deltaError, self.weight.T)
    # print(f"dE/dOutputBeforeLayer = {dError_dOutputBeforeLayer}")
    return dError_dOutputBeforeLayer

  def resetDelta(self):
    self.deltaWeight = None

  def updateWeightBias(self, learningRate, momentum):
    for i in range (len(self.weight)):
      self.weight[i][0] = self.weight[0] - ( (momentum * self.weight[i][0]) + (learningRate * self.deltaWeight[i][0] * self.input[i]) )
    self.bias[0] = self.bias[0] - ( (momentum * self.bias[0]) + (learningRate * self.deltaBias[0]) )
    self.resetDelta()

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