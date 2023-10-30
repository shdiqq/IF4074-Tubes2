import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu, sigmoid, linear

class DenseLayer():
  def __init__(self, numUnit, activationFunctionName, weight = None, bias = None):
    self.numUnit = numUnit
    self.activationFunctionName = activationFunctionName
    self.weight = weight
    self.deltaWeight = None
    self.bias = bias
    self.deltaBias = None
    self.inputData = None
    self.output = None

  def forward(self, inputData):
    self.inputData = inputData
    numFeatures = np.prod(inputData.shape)

    if (self.weight is None) :
      self.weight = np.random.randn(numFeatures, self.numUnit)
      self.weight = np.clip(self.weight, -1, 1)
    if (self.bias is None) :
      self.bias = np.zeros((self.numUnit,))

    output = np.dot(inputData, self.weight) + self.bias
    if (self.activationFunctionName.lower() == 'relu'):
      output = relu(output)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      output = sigmoid(output)
    elif (self.activationFunctionName.lower() == 'linear'):
      output = linear(output)

    self.output = output
    return output
  
  def backward(self, dError_dOutput):
    if (self.deltaWeight is None) :
      self.deltaWeight = np.zeros(self.weight.shape)
    if (self.deltaBias is None) :
      self.deltaBias = np.zeros((self.numUnit,))

    if (self.activationFunctionName.lower() == 'relu'):
      dOutput_dNet = relu(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'sigmoid'):
      dOutput_dNet = sigmoid(self.output, derivative=True)
    elif (self.activationFunctionName.lower() == 'linear'):
      dOutput_dNet = linear(self.output, derivative=True)

    deltaError = dError_dOutput * dOutput_dNet

    for i in range (len(self.deltaWeight)):
      self.deltaWeight[i] += self.inputData[i] * deltaError
    self.deltaBias += deltaError

    dError_dOutputBeforeLayer = np.dot(deltaError, self.weight.T)
    return dError_dOutputBeforeLayer

  def updateWeightBias(self, learningRate):
    for i in range (len(self.weight)):
      self.weight[i] = self.weight[i] - ( learningRate * self.deltaWeight[i] )
    self.bias = self.bias - ( learningRate * self.deltaBias )

    self.deltaWeight = None
    self.deltaBias = None
  
  def getData(self):
    return [
      {
        'type': 'dense',
        'params': {
          'numUnit': self.numUnit,
          'activationFunctionName': self.activationFunctionName,
          'weight': self.weight.tolist(),
          'bias': self.bias.tolist()
        }
      }
    ]

### TESTING ###
if __name__ == "__main__":
  matrix = np.array([118, 102])
  denseLayer = DenseLayer(numUnit = 2, activationFunctionName = 'relu', weight=np.array([[1, 2],[3, -4]]))
  newMatrix = denseLayer.forward(matrix)
  dE_dO = np.array([7.99E-02, 1.01E-02])
  dE_dOBeforeLayer = denseLayer.backward(dE_dO)
  expectedOutput = [[7.99E-02, 2.397E-01]]
  print(dE_dOBeforeLayer == expectedOutput)
  denseLayer.updateWeightBias(0.1, 1)