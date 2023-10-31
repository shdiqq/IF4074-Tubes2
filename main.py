import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from function.generateImage import *
from function.toCategorical import *
from function.prepareData import *
from function.loss import *

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
  # print("TUBES 1")
  # objectLabelDictionary = {
  #   0: 'bear',
  #   1: 'panda'
  # }
  # dataInput, dataInputLabel = generateImage()
  # dataInputLabel = toCategorical(dataInputLabel, 1)

  # # # Melakukan pembelajaran dengan skema split train 90% dan test 10%, dan menampilkan kinerja serta confusion matrixnya
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

  # cnn.fit(features = X_train, target = y_train, batchSize = 5, epoch = 1, learningRate = 0.5)

  # save_model_choice = input("Do you want to save this model? (yes/no): ").lower()
  # if save_model_choice == 'yes':
  #   model_name_to_save = input("Enter the name to save this model: ")
  #   cnn.saveModel(model_name_to_save)

  # print("Melakukan prediksi")
  # output = cnn.predict(X_test)

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

  #   cnnKfold.fit(features = X_train, target = y_train, batchSize = 5, epoch = 1, learningRate = 0.5)
  #   output = cnnKfold.predict(X_test)
    
  #   accuracy = metrics.accuracy_score(y_test, output)
  #   print("\nAccuracy:", accuracy)
  #   print("Confusion matrix:\n", metrics.confusion_matrix(y_test, output), "\n")
  #   if accuracy > best_accuracy:
  #       best_accuracy = accuracy
  #   i = i + 1
        
  # print("\nBest Accuracy:", best_accuracy)

  print("TUBES 2")
  # Read Data
  train_path = './data/train/Train_stock_market.csv'
  test_path = './data/test/Test_stock_market.csv'
  df_train = pd.read_csv(train_path, index_col='Date', parse_dates=['Date'])
  df_test = pd.read_csv(test_path, parse_dates=['Date'])

  # Scaling the training set
  scaler = MinMaxScaler(feature_range = (0, 1))
  scaled_data = scaler.fit_transform(df_train[["Low", "Open", "Volume", "High", "Close"]].values)

  # First Model
  def firstModel(inputShape):
      model = Sequential()
      model.addLayer(LSTMLayer(inputSize=(inputShape), nCells=20, returnSequences=False))
      model.addLayer(DenseLayer(numUnit=5, activationFunctionName = 'linear'))
      return model

  # Define Time Steps List & Predictions
  time_steps_list = [10]
  predictions = {}

  for time_steps in time_steps_list:
    # Prepare train data for the specific time_steps
    X_train, y_train = prepare_data(scaled_data, time_steps)

    # Create and train the model for the specific time_steps
    model = firstModel(X_train[0].shape)
    model.compile(loss="rmse")
    model.fitForwardOnly(features = X_train, target = y_train, epoch = 5)
    model.saveModel(f'model_{time_steps}')
    
    # Predict the test data
    test_samples = len(df_test)
    input_seq = scaled_data[-time_steps:].tolist()
    future_preds = []

    for i in range(test_samples):
      arrpred = input_seq[-time_steps:]
      pred = model.predict(np.array([input_seq[-time_steps:]]))
      future_preds.append(pred)
      input_seq.append(pred)
    
    future_preds = scaler.inverse_transform(future_preds)
    predictions[time_steps] = future_preds

  ## **Convert to CSV**
  last_time_step = list(predictions.keys())[-1]
  last_predictions = predictions[last_time_step]
  dates = df_test['Date'].tolist()
  low_prices = [f"{last_predictions[i][0]:.2f}" for i in range(len(dates))]
  open_prices = [f"{last_predictions[i][1]:.2f}" for i in range(len(dates))]
  volume_prices = [f"{last_predictions[i][2]:.2f}" for i in range(len(dates))]
  high_prices = [f"{last_predictions[i][3]:.2f}" for i in range(len(dates))]
  close_prices = [f"{last_predictions[i][4]:.2f}" for i in range(len(dates))]

  filePredict = pd.DataFrame({
      'Date': dates,
      'Open': open_prices,
      'High': high_prices,
      'Low': low_prices,
      'Close': close_prices,
      'Volume': volume_prices,
  })
  filePredict.to_csv('./data/predict/predict.csv', index=False)
  filePredict.head()