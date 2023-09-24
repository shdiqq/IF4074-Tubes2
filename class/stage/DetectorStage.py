import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu

class DetectorStage():
  def __init__(self):
    pass

  def forward(self, inputData):
    inputHeight, inputWidth, inputDepth = inputData.shape
    output = np.zeros([inputHeight, inputWidth, inputDepth], dtype=np.double)

    for i in range (inputDepth) :
       output[:, :, i] = relu(inputData[:, :, i], False)
    
    return output

