from function.generateImage import *

import os
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from CNN import CNN
from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer

def accuracy(dataInputLabel, dataOutputLabel):
  score = 0
  n = len(dataInputLabel)
  for i in range(n):
      if (dataInputLabel[i] == dataOutputLabel[i]) :
          score = score + 1
  final = float(score/n)
  return final

if __name__ == "__main__":
  objectLabelDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataInputLabel = generateImage()

  cnn = CNN()
  cnn.addLayer(ConvolutionalLayer(inputSize=dataInput[0].shape, filterSize = 4, numFilter = 3, mode = 'max', padding = 0, stride = 4))
  cnn.addLayer(ConvolutionalLayer(filterSize = 4, numFilter = 6, mode = 'average', padding = 0, stride = 1))
  cnn.addLayer(FlattenLayer())
  cnn.addLayer(DenseLayer(numUnit = 300, activationFunctionName = 'relu'))
  cnn.addLayer(DenseLayer(numUnit = 150, activationFunctionName = 'relu'))
  cnn.addLayer(DenseLayer(numUnit = 50, activationFunctionName = 'relu'))
  cnn.addLayer(DenseLayer(numUnit = 25, activationFunctionName = 'relu'))
  cnn.addLayer(DenseLayer(numUnit = 5, activationFunctionName = 'relu'))
  cnn.addLayer(DenseLayer(numUnit = 1, activationFunctionName = 'sigmoid'))
  dataOutputLabel = cnn.predict(dataInput)
  print("The final accuracy score is", accuracy(dataInputLabel, dataOutputLabel))