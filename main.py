from function.generateImage import *
from function.toCategorilcal import *

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
  dataInputLabel = toCategorical(dataInputLabel, 1)

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
  """ Can't be used yet
  # dataOutputLabel = cnn.predict(features=dataInput, target=dataInputLabel, batchSize=5, epoch=10, learningRate=0.5)
  # print("The final accuracy score is", accuracy(dataInputLabel, dataOutputLabel))
  """
  # Backward propagation experiment
  print("Proses forward propagation")
  output = cnn.forward(dataInput[0])
  print(f"Nilai output yang diperoleh dari forward propagation = {output}")
  print(f"Nilai target = {dataInputLabel[0]}")
  loss = cnn.calculateLoss(output, dataInputLabel[0])
  print(f"Nilai loss = {loss}")
  derivativeError = cnn.calculateDerivativeError(output, dataInputLabel[0])
  print("=====")
  print("Proses backward propagation")
  layerDelta = cnn.layers[-1].backward(derivativeError)
  # print(layerDelta)
  # print("=====")
  layerDelta = cnn.layers[-2].backward(layerDelta)
  # print(layerDelta)
  # print("=====")
  layerDelta = cnn.layers[-3].backward(layerDelta)
  # print(layerDelta)
  # print("=====")
  layerDelta = cnn.layers[-4].backward(layerDelta)
  # print(layerDelta)
  # print("=====")
  layerDelta = cnn.layers[-5].backward(layerDelta)
  # print(layerDelta)
  # print("=====")
  layerDelta = cnn.layers[-6].backward(layerDelta)
  # print(layerDelta)
  # print("=====")
  layerDeltaFlatten = cnn.layers[-7].backward(layerDelta)
  # print(layerDeltaFlatten)
  # print("=====")
  layerDeltaPoolingStage = cnn.layers[-8].backward(layerDeltaFlatten)
  # print(layerDeltaPoolingStage)
  # print("=====")