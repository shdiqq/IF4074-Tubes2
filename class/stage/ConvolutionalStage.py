import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from spatialSize import spatialSize

class ConvolutionalStage():
  def __init__(self, filterSize, numFilter, padding = 0, stride = 1, inputSize = None):
    self.inputSize = inputSize
    self.filterSize = filterSize
    self.numFilter = numFilter
    self.padding = padding
    self.stride = stride
    self.bias = np.zeros((numFilter,))
    if (self.inputSize is None) :
      self.kernel = None
    else :
      self.kernel = np.random.randn(self.numFilter, self.filterSize, self.filterSize, self.inputSize[2])

  def forward(self, inputData):
    inputHeight, inputWidth, inputDepth = inputData.shape
    outputHeight, outputWidth = spatialSize(inputHeight, inputWidth, self.filterSize, self.padding, self.stride)
    featureMap = np.zeros((outputHeight, outputWidth, self.numFilter))

    if (self.kernel is None) :
      self.kernel = np.random.randn(self.numFilter, self.filterSize, self.filterSize, inputDepth)

    for i in range(self.numFilter) :
      for row in range(0, inputData.shape[0] - self.filterSize + 1, self.stride) :
        for col in range(0, inputData.shape[1] - self.filterSize + 1, self.stride) :
          inputPatch = inputData[row : row + self.filterSize, col : col + self.filterSize, :]
          featureMap[row // self.stride, col // self.stride, i] = np.sum(inputPatch * self.kernel[i]) + self.bias[i]

    return featureMap
  
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
  print(matrix[0].shape) #Expect output (3,3,3)
  print("=====")
  convolutionalStage = ConvolutionalStage(inputSize = matrix[0].shape, filterSize = 2, numFilter = 3, padding = 1, stride = 1)
  newMatrix = convolutionalStage.forward(matrix[0])
  print(newMatrix.shape) #Expect output (4,4,3)
  print("=====")
  convolutionalStage = ConvolutionalStage(filterSize = 3, numFilter = 4, padding = 0, stride = 1)
  newMatrix2 = convolutionalStage.forward(newMatrix)
  print(newMatrix2.shape) #Expect output (2,2,4)