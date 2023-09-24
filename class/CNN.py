import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'function')
sys.path.append(mymodule_dir)

from loss import sse, crossEntropy

from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer

class CNN():
  def __init__(self):
    self.layers = []

  def addLayer(self, layer):
    self.layers.append(layer)

  def forward(self, dataInput):
    output = dataInput
    for i in range(len(self.layers)) :
      output = self.layers[i].forward(output)
    return(output)
  
  def backward(self, dError):
    deltaError = dError
    for i in reversed(range(len(self.layers))):
        deltaError = self.layers[i].backward(deltaError)
    return deltaError
  
  def calculateLoss(self, output, target):
    if (self.layers[-1].activationFunctionName.lower() == 'softmax') :
      if (len(target) == 1):
        loss = crossEntropy(target[0], output[0])
      else :
        index = target.index(1)
        loss = crossEntropy(target[index], output[index])
    else :
      if (len(target) == 1):
        loss = sse(target[0], output[0])
      else :
        index = target.index(1)
        loss = sse(target[index], output[index])
    return loss
  
  def calculateDerivativeError(self, output, target):
    derivativeError = []
    if (self.layers[-1].activationFunctionName.lower() == 'softmax') :
      for i in range (len(output)):
        derivativeError.append(crossEntropy(target[i], output[i], derivative=True))
    else :
      for i in range (len(output)):
        derivativeError.append(sse(target[i], output[i], derivative=True))
    return derivativeError

  def predict(self, features, target, batchSize, epoch, learningRate, momentum=1): # Can't be used yet
    labelOutput = []
    labelTarget = []
    for i in range(epoch):
      print("Epoch: ", i+1)
      sumLoss = 0
      for j in range(len(features)):
        print("Proses forward propagation")
        output = self.forward(features[j])
        sumLoss += self.calculateLoss(output, target[j])
        deltaError = np.array([target[j] - output[0]]) * -1
        print("Proses backward propagation")
        deltaError = self.backward(deltaError)
        print("Proses update weight")
    return(labelOutput)


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
  cnn = CNN()
  cnn.addLayer(ConvolutionalLayer(inputSize=matrix[0].shape, filterSize = 2, numFilter = 3, mode = 'max', padding = 1, stride = 1))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(ConvolutionalLayer(filterSize = 2, numFilter = 6, mode = 'average', padding = 1, stride = 1))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(FlattenLayer())
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(DenseLayer(numUnit = 16, activationFunctionName = 'relu'))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(DenseLayer(numUnit = 8, activationFunctionName = 'relu'))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(DenseLayer(numUnit = 4, activationFunctionName = 'relu'))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(DenseLayer(numUnit = 2, activationFunctionName = 'relu'))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  cnn.addLayer(DenseLayer(numUnit = 1, activationFunctionName = 'relu'))
  print(cnn.forward(matrix[0]).shape)
  print("=====")
  print(cnn.predict(matrix))