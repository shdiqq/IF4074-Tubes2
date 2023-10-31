import os
import sys
import json
import numpy as np
from sklearn import metrics

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'function')
sys.path.append(mymodule_dir)

from loss import sumSquareError, crossEntropy, meanSquareError, rootMeanSquareError

from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer
from layer.LSTMLayer import LSTMLayer

class Sequential():
  def __init__(self):
    self.layers = []
    self.loss = ""

  def addLayer(self, layer):
    self.layers.append(layer)

  def compile(self, loss):
    self.loss = loss

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
  
  def updateWeightBias(self, learningRate):
    for i in reversed(range(len(self.layers))):
      self.layers[i].updateWeightBias(learningRate)
  
  def calculateLoss(self, output, target):
    loss = 0
    if (self.loss == "crossEntropy") :
      for i in range (len(target)):
        loss = loss + crossEntropy(target[i], output[i])
    elif (self.loss == "mse"):
      for i in range (len(target)):
        loss = loss + meanSquareError(target[i], output[i])
    elif (self.loss == "rmse"):
      for i in range (len(target)):
        loss = loss + rootMeanSquareError(target[i], output[i])
    else :
      for i in range (len(target)):
        loss = loss + sumSquareError(target[i], output[i])
    return loss/len(target)
  
  def calculateDerivativeError(self, output, target):
    derivativeError = []
    if (self.loss == "crossEntropy") :
      for i in range (len(output)):
        derivativeError.append(crossEntropy(target[i], output[i], derivative=True))
    else :
      for i in range (len(output)):
        derivativeError.append(sumSquareError(target[i], output[i], derivative=True))
    return derivativeError

  def fit(self, features, target, batchSize, epoch, learningRate):
    labelOutput = np.array([])
    labelTarget = np.array([])

    print("  Epoch   |     Loss    |   Accuracy   |      Output -> Target")
    print("=========================================================================================")
    for i in range(epoch):
      sumLoss = 0
      idxReadInputData = 0
      while ( idxReadInputData != len(features) ):
        for j in range (batchSize):
          # Proses Forward Propagation
          output = self.forward(features[idxReadInputData])
          labelOutput = np.rint(np.append(labelOutput, output))
          labelTarget = np.append(labelTarget, target[idxReadInputData])
          sumLoss += self.calculateLoss(output, target[idxReadInputData])

          # Proses Backward Propagation
          derivativeError = self.calculateDerivativeError(output, target[idxReadInputData])
          derivativeError = self.backward(derivativeError)

          idxReadInputData = idxReadInputData + 1

        # Proses Update Weight dan Bias
        self.updateWeightBias(learningRate)

      avgLoss = sumLoss/len(features)

      print(f"    {i+1}     |   {avgLoss:.5f}   |     {metrics.accuracy_score(labelTarget, labelOutput):.2f}     |   {labelOutput}  ->  {labelTarget}   ")
      print("-----------------------------------------------------------------------------------------")

  def fitForwardOnly(self, features, target, epoch):
    labelOutput = np.array([])
    labelTarget = np.array([])

    print("  Epoch   |     Loss    |")
    print("========================")
    for i in range(epoch):
      sumLoss = 0
      idxReadInputData = 0
      while ( idxReadInputData != len(features) ):
        # Proses Forward Propagation
        output = self.forward(features[idxReadInputData])
        labelOutput = np.rint(np.append(labelOutput, output))
        labelTarget = np.append(labelTarget, target[idxReadInputData])
        sumLoss += self.calculateLoss(output, target[idxReadInputData])

        idxReadInputData = idxReadInputData + 1

      avgLoss = sumLoss/len(features)

      print(f"    {i+1}     |   {avgLoss:.5f}   |")
      print("-------------------------")

  def predict(self, inputTest):
    output = np.array([])
    for data in inputTest:
      outputForward = self.forward(data)
      output = np.append(output, outputForward)
    return output


  def saveModel(self, filename):
    file = open(f'./model/{filename}.json', 'w')
    data = []

    for i in range(len(self.layers)):
      data += self.layers[i].getData()

    file.write(json.dumps(data, indent=2))
    file.close()

  def loadModel(self, filename):
    file = open(f'./model/{filename}.json', 'r')
    data = json.loads(file.read())

    for i in range(len(data)):
      if (data[i]['type'] == 'convolutional'):
        self.addLayer(ConvolutionalLayer(
          filterSize = data[i]['params']['filterSize'],
          numFilter = data[i]['params']['numFilter'],
          mode = data[i]['params']['mode'],
          padding = data[i]['params']['padding'],
          stride = data[i]['params']['stride'],
          kernel = np.array(data[i]['params']['kernel']),
          bias = np.array(data[i]['params']['bias'])
        ))
      elif (data[i]['type'] == 'flatten'):
        self.addLayer(FlattenLayer())
      elif (data[i]['type'] == 'dense'):
        self.addLayer(DenseLayer(
          numUnit = data[i]['params']['numUnit'],
          activationFunctionName = data[i]['params']['activationFunctionName'],
          weight = np.array(data[i]['params']['weight']),
          bias = np.array(data[i]['params']['bias'])
        ))
    file.close()

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
  cnn = Sequential()
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