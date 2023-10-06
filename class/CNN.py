import os
import sys
import numpy as np
from sklearn import metrics

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
  
  def updateWeightBias(self, learningRate, momentum):
    for i in reversed(range(len(self.layers))):
      self.layers[i].updateWeightBias(learningRate, momentum)
  
  def calculateLoss(self, output, target):
    if (self.layers[-1].activationFunctionName.lower() == 'softmax') :
      if (len(target) == 1):
        loss = crossEntropy(target[0], output[0])
      else :
        index = np.where(target == 1)[0][0]
        loss = crossEntropy(target[index], output[index])
    else :
      if (len(target) == 1):
        loss = sse(target[0], output[0])
      else :
        index = np.where(target == 1)[0][0] 
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

  def predict(self, features, target, batchSize, epoch, learningRate, momentum=1):
    labelOutput = np.array([])
    labelTarget = np.array([])

    for i in range(epoch):
      print("Epoch ke-", i + 1)
      sumLoss = 0
      for j in range(batchSize):
        # if (j + 1) % 5 == 0:
        #     print("Batch size ke- 5")
        # else:
        #   print("Batch size ke-", (j + 1) % batchSize)
        currentIndex = (batchSize * i + j) % len(features)
        # print("Proses forward propagation")
        output = self.forward(features[currentIndex])
        labelOutput = np.rint(np.append(labelOutput, output))
        labelTarget = np.append(labelTarget, target[currentIndex])
        sumLoss += self.calculateLoss(output, target[currentIndex])

        # print("Proses backward propagation")
        derivativeError = self.calculateDerivativeError(output, target[currentIndex])
        derivativeError = self.backward(derivativeError)

      # print("Proses update weight")
      self.updateWeightBias(learningRate, momentum)

      avgLoss = sumLoss/len(features)
      print('Nilai Loss adalah ', avgLoss)
      print("Output yang diperoleh ", labelOutput)
      print("Target yang diharapkan ", labelTarget)
      print('Nilai Akurasi adalah ', metrics.accuracy_score(labelTarget, labelOutput))


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