import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu

class DetectorStage():
  def __init__(self):
    self.inputData = None
    self.output = None

  def forward(self, inputData):
    self.inputData = inputData
    inputHeight, inputWidth, inputDepth = inputData.shape
    output = np.zeros([inputHeight, inputWidth, inputDepth], dtype=np.double)

    for i in range (inputDepth):
       output[:, :, i] = relu(inputData[:, :, i], False)
    
    self.output = output
    return output

  def backward(self, dError_dOutput):
    inputHeight, inputWidth, inputDepth = self.inputData.shape
    dError_dOutputBeforeLayer = np.zeros_like(self.inputData, dtype=np.double)

    for i in range (inputDepth):
      for row in range (inputHeight):
        for col in range (inputWidth):
          if (self.inputData[row, col, i] > 0):
            dError_dOutputBeforeLayer[row, col, i] = dError_dOutput[row, col, i]
          else:
            dError_dOutputBeforeLayer[row, col, i] = 0

    return dError_dOutputBeforeLayer
  
### TESTING ###
if __name__ == '__main__':
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
  detectorStage = DetectorStage()
  print(detectorStage.forward(matrix[0]))
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
  print(detectorStage.backward(dE_dO))
