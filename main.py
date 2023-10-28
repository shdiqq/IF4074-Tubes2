import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from function.generateImage import *
from function.toCategorical import *

import os
import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from Sequential import Sequential
from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer
from layer.LSTMLayer import LSTMLayer

if __name__ == "__main__":
  """ TUBES 1 """
  # objectLabelDictionary = {
  #   0: 'bear',
  #   1: 'panda'
  # }
  # dataInput, dataInputLabel = generateImage()
  # dataInputLabel = toCategorical(dataInputLabel, 1)

  # # Melakukan pembelajaran dengan skema split train 90% dan test 10%, dan menampilkan kinerja serta confusion matrixnya
  # print("1. Implement training using 90% train and 10% test split, and show its performance and confusion matrix\n")
  # X_train, X_test, y_train, y_test = train_test_split(dataInput, dataInputLabel, test_size=0.1)

  # cnn = Sequential()
  # load_model_choice = input("Do you want to load a pre-existing model? (yes/no): ").lower()
  # if load_model_choice == 'yes':
  #   while True:
  #     model_name = input("Enter the model name to load: ")
  #     filename = os.path.join(script_dir, 'model', model_name + '.json')
  #     if os.path.exists(filename):
  #       break
  #     else:
  #       print("Model not found. Please try again.")
  #       continue
  #   print()
  #   cnn.loadModel(model_name)
  # else:
  #   cnn.addLayer(ConvolutionalLayer(inputSize=dataInput[0].shape, filterSize = 4, numFilter = 3, mode = 'max', padding = 0, stride = 4))
  #   cnn.addLayer(ConvolutionalLayer(filterSize = 4, numFilter = 6, mode = 'average', padding = 0, stride = 1))
  #   cnn.addLayer(FlattenLayer())
  #   cnn.addLayer(DenseLayer(numUnit = 300, activationFunctionName = 'relu'))
  #   cnn.addLayer(DenseLayer(numUnit = 150, activationFunctionName = 'relu'))
  #   cnn.addLayer(DenseLayer(numUnit = 50, activationFunctionName = 'relu'))
  #   cnn.addLayer(DenseLayer(numUnit = 25, activationFunctionName = 'relu'))
  #   cnn.addLayer(DenseLayer(numUnit = 5, activationFunctionName = 'relu'))
  #   cnn.addLayer(DenseLayer(numUnit = 1, activationFunctionName = 'sigmoid'))

  # cnn.predict(features = X_train, target = y_train, batchSize = 5, epoch = 10, learningRate = 0.5)
  
  # save_model_choice = input("Do you want to save this model? (yes/no): ").lower()
  # if save_model_choice == 'yes':
  #   model_name_to_save = input("Enter the name to save this model: ")
  #   cnn.saveModel(model_name_to_save)

  # output = np.array([])
  # for data in X_test:
  #   forwardCNN = cnn.forward(data)
  #   output = np.append(output, np.rint(forwardCNN))

  # print("\nTarget:", y_test)
  # print("Predicted:", output)
  # print("Accuracy:", metrics.accuracy_score(y_test, output))
  # print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output))

  # # Melakukan pembelajaran dengan skema 10-fold cross validation, dan menampilkan kinerjanya.
  # print("\n2. Implement training using 10-fold cross validation, and show its performance.\n")
  # kf = KFold(n_splits=10,shuffle=True)
  # best_accuracy = 0
  # best_model = None
  # i = 1
  # for train_index, test_index in kf.split(dataInput):
  #   print("SPLIT - ", i)
  #   X_train, X_test = dataInput[train_index], dataInput[test_index]
  #   dataInputLabel = np.array(dataInputLabel)
  #   y_train, y_test = dataInputLabel[train_index], dataInputLabel[test_index]

  #   cnnKfold = Sequential()
  #   print("Reading (load) model from external file (model.json)")
  #   cnnKfold.loadModel('model')

  #   output = np.array([])
  #   for data in X_test:
  #       forward_cnn = cnnKfold.forward(data)
  #       output = np.append(output, np.rint(forward_cnn))
    
  #   cnnKfold.predict(features = X_train, target = y_train, batchSize = 5, epoch = 10, learningRate = 0.5)
  #   accuracy = metrics.accuracy_score(y_test, output)
  #   print("\nAccuracy:", accuracy)
  #   print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output), "\n")
  #   if accuracy > best_accuracy:
  #       best_accuracy = accuracy
  #   i = i + 1
        
  # print("\nBest Accuracy:", best_accuracy)

  """ TUBES 2 """
  train_path = './data/train/Train_stock_market.csv'
  test_path = './data/test/Test_stock_market.csv'

  df_train = pd.read_csv(train_path, infer_datetime_format=True)
  df_test = pd.read_csv(test_path, infer_datetime_format=True)
  dataset = df_train.loc[:, ['Close']].values
  dataset = dataset.reshape(-1, 1)
  scaler = MinMaxScaler(feature_range = (0, 1))
  data_scaled = scaler.fit_transform(dataset)

  X_train = []
  y_train = []
  time_step = 10
  for i in range(len(data_scaled) - time_step - 1):
      a = data_scaled[i:(i + time_step), 0]
      X_train.append(a)
      y_train.append(data_scaled[i + time_step, 0])
  X_train = np.array(X_train)
  y_train = np.array(y_train)
  print("trainX shape: {}\ntrainY shape: {}". format(X_train.shape, y_train.shape))
  print(X_train[0])