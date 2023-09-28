import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu, sigmoid, linear

class DenseLayer():
  def __init__(self, numUnit, activationFunctionName, weight = None):
    self.numUnit = numUnit
    self.activationFunctionName = activationFunctionName
    self.weight = weight
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
    if (self.deltaBias is None) :
      self.deltaBias = np.zeros((self.numUnit,))

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
    if (self.activationFunctionName.lower() == 'relu'):
      dOutput_dNet = relu(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      dOutput_dNet = sigmoid(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'linear'):
      dOutput_dNet = linear(self.output, derivative=True)
    # elif (self.activationFunctionName.lower() == 'softmax'):
    #   derivativeValues = softmax(self.output, derivative=True)

    deltaError = dError_dOutput * dOutput_dNet

    for i in range (len(self.deltaWeight)):
      self.deltaWeight[i] += self.input[i] * deltaError
    self.deltaBias += deltaError

    dError_dOutputBeforeLayer = np.dot(deltaError, self.weight.T)
    return dError_dOutputBeforeLayer

  def resetDelta(self):
    self.deltaWeight = None
    self.deltaBias = None

  def updateWeightBias(self, learningRate, momentum):
    for i in range (len(self.weight)):
      self.weight[i][0] = self.weight[0] - ( (momentum * self.weight[i][0]) + (learningRate * self.deltaWeight[i][0] * self.input[i]) )
    self.bias = self.bias - ( (momentum * self.bias) + (learningRate * self.deltaBias) )
    self.resetDelta()

### TESTING ###
if __name__ == "__main__":
  matrix = np.array([118, 102])
  denseLayer = DenseLayer(numUnit = 2, activationFunctionName = 'relu', weight=np.array([[1, 2],[3, -4]]))
  newMatrix = denseLayer.forward(matrix)
  dE_dO = np.array([7.99E-02, 1.01E-02])
  dE_dOBeforeLayer = denseLayer.backward(dE_dO)
  expectedOutput = [[7.99E-02, 2.397E-01]]
  print(dE_dOBeforeLayer == expectedOutput)