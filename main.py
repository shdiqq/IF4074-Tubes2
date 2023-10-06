from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from function.generateImage import *
from function.toCategorical import *

import os
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from CNN import CNN
from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer

if __name__ == "__main__":
  objectLabelDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataInputLabel = generateImage()
  dataInputLabel = toCategorical(dataInputLabel, 1)

  # Melakukan pembelajaran dengan skema split train 90% dan test 10%, dan menampilkan kinerja serta confusion matrixnya
  print("1.Implementasi dalam melakukan pembelajaran dengan skema split train 90% dan test 10%, dan menampilkan kinerja serta confusion matrixnya")
  X_train, X_test, y_train, y_test = train_test_split(dataInput, dataInputLabel, test_size=0.1)

  cnn = CNN()
  # cnn.loadModel('model2')
  # # cnn.addLayer(ConvolutionalLayer(inputSize=dataInput[0].shape, filterSize = 4, numFilter = 3, mode = 'max', padding = 0, stride = 4))
  # # cnn.addLayer(ConvolutionalLayer(filterSize = 4, numFilter = 6, mode = 'average', padding = 0, stride = 1))
  # # cnn.addLayer(FlattenLayer())
  # # cnn.addLayer(DenseLayer(numUnit = 300, activationFunctionName = 'relu'))
  # # cnn.addLayer(DenseLayer(numUnit = 150, activationFunctionName = 'relu'))
  # # cnn.addLayer(DenseLayer(numUnit = 50, activationFunctionName = 'relu'))
  # # cnn.addLayer(DenseLayer(numUnit = 25, activationFunctionName = 'relu'))
  # # cnn.addLayer(DenseLayer(numUnit = 5, activationFunctionName = 'relu'))
  # # cnn.addLayer(DenseLayer(numUnit = 1, activationFunctionName = 'sigmoid'))

  # cnn.predict(features = X_train, target = y_train, batchSize = 5, epoch = 10, learningRate = 0.5)

  output = np.array([])
  for data in X_test:
    forwardCNN = cnn.forward(data)
    output = np.append(output, np.rint(forwardCNN))

  print("\nTarget:", y_test)
  print("Predicted:", output)
  print("Accuracy:", metrics.accuracy_score(y_test, output))
  print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output))

  # Melakukan pembelajaran dengan skema 10-fold cross validation, dan menampilkan kinerjanya.
  print("\n2.Implementasi dalam pembelajaran dengan skema 10-fold cross validation, dan menampilkan kinerjanya.")
  kf = KFold(n_splits=10,shuffle=True)
  best_accuracy = 0
  best_model = None
  i = 1
  for train_index, test_index in kf.split(dataInput):
    print("Split ke-", i)
    X_train, X_test = dataInput[train_index], dataInput[test_index]
    dataInputLabel = np.array(dataInputLabel)
    y_train, y_test = dataInputLabel[train_index], dataInputLabel[test_index]

    cnnKfold = CNN()
    cnnKfold.addLayer(ConvolutionalLayer(inputSize=dataInput[0].shape, filterSize = 4, numFilter = 3, mode = 'max', padding = 0, stride = 4))
    cnnKfold.addLayer(ConvolutionalLayer(filterSize = 4, numFilter = 6, mode = 'average', padding = 0, stride = 1))
    cnnKfold.addLayer(FlattenLayer())
    cnnKfold.addLayer(DenseLayer(numUnit = 300, activationFunctionName = 'relu'))
    cnnKfold.addLayer(DenseLayer(numUnit = 150, activationFunctionName = 'relu'))
    cnnKfold.addLayer(DenseLayer(numUnit = 50, activationFunctionName = 'relu'))
    cnnKfold.addLayer(DenseLayer(numUnit = 25, activationFunctionName = 'relu'))
    cnnKfold.addLayer(DenseLayer(numUnit = 5, activationFunctionName = 'relu'))
    cnnKfold.addLayer(DenseLayer(numUnit = 1, activationFunctionName = 'sigmoid'))
    cnnKfold.predict(features = X_train, target = y_train, batchSize = 5, epoch = 3, learningRate = 0.1)

    output = np.array([])
    for data in X_test:
        forward_cnn = cnnKfold.forward(data)
        output = np.append(output, np.rint(forward_cnn))
    
    accuracy = metrics.accuracy_score(y_test, output)
    print("\nAccuracy:", accuracy)
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output), "\n")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
    i = i + 1
        
  print("\nBest Accuracy:", best_accuracy)

