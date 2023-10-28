import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'stage')
sys.path.append(mymodule_dir)

from ConvolutionalStage import ConvolutionalStage
from DetectorStage import DetectorStage
from PoolingStage import PoolingStage

class ConvolutionalLayer():
  def __init__(self, filterSize, numFilter, mode, padding = 0, stride = 1, inputSize = None, kernel = None, bias = None):
    self.filterSize = filterSize
    self.numFilter = numFilter
    self.mode = mode
    self.padding = padding
    self.stride = stride
    self.inputSize = inputSize
    self.convolutionStage = ConvolutionalStage(filterSize, numFilter, padding, stride, inputSize, kernel, bias)
    self.detectorStage = DetectorStage()
    self.poolingStage = PoolingStage(filterSize, stride, mode)

  def forward(self, inputData):
    featureMap = self.convolutionStage.forward(inputData)
    outputDetector = self.detectorStage.forward(featureMap)
    outputPooling = self.poolingStage.forward(outputDetector)
    return outputPooling
  
  def backward(self, dError_dOutput):
    delta_pooling = self.poolingStage.backward(dError_dOutput)
    delta_detector = self.detectorStage.backward(delta_pooling)
    delta_convolutional = self.convolutionStage.backward(delta_detector)
    return delta_convolutional
  
  def updateWeightBias(self, learningRate, momentum):
    self.convolutionStage.updateKernelBias(learningRate, momentum)

  def getData(self):
    pooling_stage = "max_pooling" if self.mode == "max" else "average_pooling" 
    return [
      {
        'type': 'convolutional',
        'params': {
          'filterSize': self.filterSize,
          'numFilter': self.numFilter,
          'padding': self.padding,
          'stride': self.stride,
          'inputSize': self.inputSize,
          'kernel': self.convolutionStage.kernel.tolist(),
          'bias': self.convolutionStage.bias.tolist(),
          'mode': self.mode
        }
      },
      {
        'type': pooling_stage,
        'params': {
          'filterSize': self.filterSize,
          'stride': self.stride,
          'mode': self.mode
        }
      }
    ]
  
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
  convolutionalLayer = ConvolutionalLayer(inputSize = matrix[0].shape, filterSize = 2, numFilter = 3, mode='max', padding = 1, stride = 1)
  newMatrix = convolutionalLayer.forward(matrix[0])
  print(newMatrix.shape)
  print("=====")
  convolutionalLayer2 = ConvolutionalLayer(filterSize = 2, numFilter = 4, mode='max', padding = 0, stride = 1)
  newMatrix1 = convolutionalLayer2.forward(newMatrix)
  print(newMatrix1.shape)
  print("=====")
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