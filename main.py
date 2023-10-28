import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from function.generateImage import *
from function.toCategorical import *
from function.prepareData import *
from function.errorCalc import *

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
    #   model.addLayer(LSTMLayer(inputSize=inputShape[2], nCells=inputShape[1]))
      # print(inputShape)
      # print(inputShape[1])
      # print(inputShape[2])
      model.addLayer(LSTMLayer(inputSize=inputShape[2], nCells=inputShape[1]))
      model.addLayer(DenseLayer(numUnit=inputShape[0], activationFunctionName = 'linear'))
      return model

  # Define Time Steps List & Predictions
  time_steps_list = [5]
  predictions = {}

  for time_steps in time_steps_list:
    # Prepare train data for the specific time_steps
    X_train, y_train = prepare_data(scaled_data, time_steps)
    print("shape", X_train.shape)

    # Create and train the model for the specific time_steps
    model = firstModel((X_train.shape))
    output = model.forward(X_train)

    # Predict the test data
    test_samples = len(df_test)
    input_seq = scaled_data[-time_steps:]
    print("X_train")
    print(X_train)
    print("input seq:", input_seq)
    future_preds = []

    print("Output\n", output)

    for i in range(test_samples):
        print("MASUK KE YANG PREDICT")
        current_pred = model.forward(np.array([input_seq]))
        future_preds.append(current_pred)
        input_seq.append(current_pred)

    future_preds = scaler.inverse_transform(future_preds)
    predictions[time_steps] = future_preds

    last_time_step = list(predictions.keys())[-1]
    last_preds = predictions[last_time_step]
    dates = df_test['Date'].to_list()
    open_prices = [f"{last_preds[i][0]:.2f}" for i in range(len(dates))]
    close_prices = [f"{last_preds[i][1]:.2f}" for i in range(len(dates))]
    high_prices = [f"{last_preds[i][2]:.2f}" for i in range(len(dates))]
    low_prices = [f"{last_preds[i][3]:.2f}" for i in range(len(dates))]
    volumes = [f"{last_preds[i][4]:.2f}" for i in range(len(dates))]

    # Create a dataframe with the predicted prices
    predicted_df = pd.DataFrame({
        'Date': np.array(dates),
        'Open': np.array(open_prices),
        'High': np.array(high_prices),
        'Low': np.array(low_prices),
        'Close': np.array(close_prices),
        'Volume': np.array(volumes)
    })

    print("predicted_df")
    print(predicted_df.head())
    print("df_test")
    print(df_test.head())

    # Calculate RMSE for each column and store the results in a dictionary
    errors = {}
    columns_to_compare = ["Open", "High", "Low", "Close", "Volume"]
    for column in columns_to_compare:
        errors[column] = root_mean_squared_error(df_test[column], predicted_df[column])

    # Display the RMSE for each column
    for column, error in errors.items():
        print(f"RMSE for {column}: {error}")