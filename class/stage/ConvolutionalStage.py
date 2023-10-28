import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from spatialSize import spatialSize

class ConvolutionalStage():
  def __init__(self, filterSize, numFilter, padding = 0, stride = 1, inputSize = None, kernel = None, bias = None):
    self.inputSize = inputSize
    self.filterSize = filterSize
    self.numFilter = numFilter
    self.padding = padding
    self.stride = stride
    self.kernel = kernel
    self.deltaKernel = None
    self.bias = bias
    self.deltaBias = None
    self.inputData = None

  def forward(self, inputData):
    self.inputData = np.pad(inputData, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
    inputHeight, inputWidth, inputDepth = inputData.shape
    outputHeight, outputWidth = spatialSize(inputHeight, inputWidth, self.filterSize, self.padding, self.stride)
    featureMap = np.zeros((outputHeight, outputWidth, self.numFilter))

    if (self.kernel is None) :
      self.kernel = np.random.randn(self.numFilter, self.filterSize, self.filterSize, inputDepth)
    if (self.bias is None) :
      self.bias = np.zeros((self.numFilter,))

    for i in range(self.numFilter) :
      for row in range(0, inputData.shape[0] - self.filterSize + 1, self.stride) :
        for col in range(0, inputData.shape[1] - self.filterSize + 1, self.stride) :
          inputPatch = inputData[row : row + self.filterSize, col : col + self.filterSize, :]
          featureMap[row // self.stride, col // self.stride, i] = np.sum(inputPatch * self.kernel[i]) + self.bias[i]

    return featureMap
  
  def backward(self, dError_dOutput):
    if (self.deltaKernel is None) :
      self.deltaKernel = np.zeros(self.kernel.shape)
    if (self.deltaBias is None) :
      self.deltaBias = np.zeros(self.bias.shape)

    # Finding dE/dK = deltaKernel
    for i in range(self.numFilter) :
      for row in range(0, dError_dOutput.shape[0] - self.filterSize + 1, self.stride) :
        for col in range(0, dError_dOutput.shape[1] - self.filterSize + 1, self.stride) :
          inputPatch = self.inputData[row : row + self.filterSize, col : col + self.filterSize, :]
          self.deltaKernel[i, :, :, :] = dError_dOutput[row, col, i] * inputPatch
    
    # Finding dE/dB = deltaBias
    for i in range(self.numFilter):
      self.deltaBias[i] = np.sum(dError_dOutput[:, :, i])

    # Finding dE/dX = dError_dOutputBefore
    dError_dOutputBefore = np.zeros_like(self.inputData, dtype=np.double)
    for i in range(self.numFilter):
        rotated_kernel = np.rot90(self.kernel[i], 2)
        for row in range(0, dError_dOutput.shape[0] - self.filterSize + 1, self.stride):
            for col in range(0, dError_dOutput.shape[1] - self.filterSize + 1, self.stride):
                dError_patch = dError_dOutput[row: row + self.filterSize, col: col + self.filterSize, i]
                # Full convolution between the rotated kernel and the dError_patch
                for k_row in range(self.filterSize):
                    for k_col in range(self.filterSize):
                        dError_dOutputBefore[row + k_row, col + k_col, :] += rotated_kernel[k_row, k_col, :] * dError_patch[k_row, k_col]
                        
    return dError_dOutputBefore
    
  def updateKernelBias(self, learningRate, momentum):
      self.kernel -= learningRate * self.deltaKernel
      self.bias -= learningRate * self.deltaBias

      self.delta_w = None
      self.delta_b = None

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
  convolutionalStage = ConvolutionalStage(inputSize = matrix[0].shape, filterSize = 3, numFilter = 3, padding = 1, stride = 1)
  newMatrix = convolutionalStage.forward(matrix[0])
  print(newMatrix) #Expect output (4,4,3)
  print("=====")
  # convolutionalStage = ConvolutionalStage(filterSize = 3, numFilter = 4, padding = 0, stride = 1)
  # newMatrix2 = convolutionalStage.forward(newMatrix)
  # print(newMatrix2.shape) #Expect output (2,2,4)
  # print("=====")
  dE_dO = np.array(
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
  dE_dX = convolutionalStage.backward(dE_dO)
  print(dE_dO)
  print("===== 1")
  print(dE_dX)