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
    self.inputData = None
    self.output = None

  def forward(self, inputData):
    self.inputData = inputData
    inputHeight, inputWidth, inputDepth = inputData.shape
    outputHeight, outputWidth = spatialSize(inputHeight, inputWidth, self.filterSize, 0, self.stride)

    output = np.zeros([outputHeight, outputWidth, inputDepth], dtype=np.double)

    for i in range(inputDepth):
      for row in range(0, inputHeight - self.filterSize + 1, self.stride) :
        for col in range(0, inputWidth - self.filterSize + 1, self.stride) :
          if (self.mode.lower() == 'max'):
            output[row // self.stride, col // self.stride, i] = self.modeMaxForward(inputData[row : row + self.filterSize, col : col + self.filterSize, i])
          else:
            output[row // self.stride, col // self.stride, i] = self.modeAverageForward(inputData[row : row + self.filterSize, col : col + self.filterSize, i])
    self.output = output
    return output

  def backward(self, dError_dOutput):
    inputHeight, inputWidth, inputDepth = self.inputData.shape
    dError_dOutputBeforeLayer = np.zeros_like(self.inputData, dtype=np.double)

    for i in range(inputDepth):
      for row in range(0, inputHeight - self.filterSize + 1, self.stride):
        for col in range(0, inputWidth - self.filterSize + 1, self.stride):
          if (self.mode.lower() == 'max'):
            listMaxIndex = self.modeMaxBackward(self.inputData[row : row + self.filterSize, col : col + self.filterSize, i], self.output[row // self.stride, col // self.stride, i])
            for maxIndex in listMaxIndex:
              dError_dOutputBeforeLayer[row : row + self.filterSize, col : col + self.filterSize, i][maxIndex] += dError_dOutput[row // self.stride, col // self.stride, i]
          else:
            dError_dOutputBeforeLayer[row : row + self.filterSize, col : col + self.filterSize, i] += self.modeAverageBackward(dError_dOutput[row // self.stride, col // self.stride, i], self.filterSize)

    return dError_dOutputBeforeLayer

  def modeMaxForward(self, inputs):
    max = np.max(inputs)
    return max

  def modeAverageForward(self, inputs):
    avg = '%.3f' % np.average(inputs)
    return avg

  def modeMaxBackward(self, inputs, outputs):
    listMaxIndex = []
    for i in range(len(inputs)) :
      for j in range(len(inputs[0])) :
        if (inputs[i][j] == outputs):
          listMaxIndex.append((i, j))
    return listMaxIndex

  def modeAverageBackward(self, dError_dOutput, filterSize):
    return dError_dOutput / (filterSize * filterSize)

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
  poolingStage = PoolingStage(filterSize = 2, stride = 1, mode='max')
  newMatrix = poolingStage.forward(matrix[0])
  print(newMatrix)
  print("===")
  dE_dO = np.array(
    [
      [
        [ 12, 12, 12 ],
        [ 12, 12, 12 ],
      ],
      [
        [ 12, 12, 12 ],
        [ 12, 12, 12 ],
      ],
    ]
  )
  eror = poolingStage.backward(dE_dO)
  expectedResultMax = np.array(
    [
      [
        [0,12,0],
        [0,0,12],
        [0,12,12],
      ],
      [
        [24,0,0],
        [0,0,0],
        [12,0,0],
      ],
      [
        [0,12,0],
        [0,12,12],
        [12,0,12],
      ]
    ]
  )
  expectedResultAverage = np.array(
    [
      [
        [3,3,3],
        [6,6,6],
        [3,3,3],
      ],
      [
        [6,6,6],
        [12,12,12],
        [6,6,6],
      ],
      [
        [3,3,3],
        [6,6,6],
        [3,3,3],
      ]
    ]
  )
  print(eror == expectedResultMax)