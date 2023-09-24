import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from spatialSize import spatialSize

class PoolingStage():
  def __init__(self, filterSize, stride, mode):
    self.filterSize = filterSize
    self.stride = stride
    self.mode = mode
    
  def forward(self, inputData):
    inputHeight, inputWidth, inputDepth = inputData.shape
    outputHeight, outputWidth = spatialSize(inputHeight, inputWidth, self.filterSize, 0, self.stride)

    output = np.zeros([outputHeight, outputWidth, inputDepth], dtype=np.double)

    for i in range(inputDepth):
      for row in range(0, inputHeight - self.filterSize + 1, self.stride) :
        for col in range(0, inputWidth - self.filterSize + 1, self.stride) :
          if (self.mode.lower() == 'max'):
            output[row // self.stride, col // self.stride, i] = self.modeMax(inputData[row : row + self.filterSize, col : col + self.filterSize, i])
          else:
            output[row // self.stride, col // self.stride, i] = self.modeAverage(inputData[row : row + self.filterSize, col : col + self.filterSize, i])

    return output

  def modeMax(self, inputs):
    max = np.max(inputs)
    return max

  def modeAverage(self,inputs):
    avg = '%.3f' % np.average(inputs)
    return avg

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
  poolingStage = PoolingStage(filterSize = 2, stride = 1, mode='max')
  newMatrix = poolingStage.forward(matrix[0])
  print(newMatrix.shape)