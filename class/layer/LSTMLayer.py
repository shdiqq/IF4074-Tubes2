import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import sigmoid

class LSTMParameter:
  def __init__(self, nAttributes, nCells):
    self.u = np.random.rand(nAttributes, nCells)
    self.u = np.clip(self.u, -1, 1)
    self.w = np.random.rand(nCells, nCells)
    self.w = np.clip(self.w, -1, 1)
    self.b = np.random.rand(nCells)
    self.b = np.clip(self.b, -1, 1)

class LSTMLayer():
  def __init__(self, inputSize, nCells, returnSequences=False):
    self.inputData = None
    self.inputSize = inputSize
    self.timeSteps = inputSize[0]
    self.nAttributes = inputSize[1]
    self.nCells = nCells
    self.returnSequences = returnSequences
    self.cellPrev = np.zeros((self.timeSteps + 1, self.nCells))
    self.hiddenPrev = np.zeros((self.timeSteps + 1, self.nCells))

    # 4 gate param
    self.forgetParam = LSTMParameter(self.nAttributes, self.nCells)
    self.inputParam = LSTMParameter(self.nAttributes, self.nCells)
    self.cellParam = LSTMParameter(self.nAttributes, self.nCells)
    self.outputParam = LSTMParameter(self.nAttributes, self.nCells)

  def forgetGate(self, timestep):
    # sigmoid(Uf * inputData + Wf * hiddenPrev + Bf)
    net_f = np.dot(self.inputData[timestep], self.forgetParam.u) + np.dot(self.hiddenPrev[timestep], self.forgetParam.w) + self.forgetParam.b
    ft = sigmoid(net_f)
    return (ft)

  def inputGate(self, timestep):
    # sigmoid(inputData * Ui + hiddenPrev * Wi + Bi)
    net_i = np.matmul(self.inputData[timestep], self.inputParam.u) + np.dot(self.hiddenPrev[timestep], self.inputParam.w) + self.inputParam.b
    it = sigmoid(net_i)

    # # tanH(inputData * Uc + hiddenPrev * Wc + Bc)
    net_candidate = np.dot(self.inputData[timestep], self.cellParam.u) + np.dot(self.hiddenPrev[timestep], self.cellParam.w) + self.cellParam.b
    candidate_t = np.tanh(net_candidate)
    return it, candidate_t

  def cellState(self, timestep, ft, it, candidate_t):
    # candidate * cellPrev + inputGate * candidate
    ct = np.multiply(ft, self.cellPrev[timestep]) + np.multiply(it, candidate_t)
    return ct

  def outputGate(self, timestep, ct):
    # sigmoid(inputData * Uo + hiddenPrev * Wo + Bo)
    net_o = np.dot(self.inputData[timestep], self.outputParam.u) + np.dot(self.hiddenPrev[timestep], self.outputParam.w) + self.outputParam.b
    ot = sigmoid(net_o)

    # output * tanH(cellState)
    ht = np.multiply(ot, np.tanh(ct))
    return ht
  
  def forward(self, inputData):
    self.inputData = inputData
    for i in range(1, self.timeSteps + 1):
      ft = self.forgetGate(i-1)
      it, candidate_t = self.inputGate(i-1)
      ct = self.cellState(i-1, ft, it, candidate_t)
      ht = self.outputGate(i-1, ct)

      self.cellPrev[i] = ct
      self.hiddenPrev[i] = ht
    
    if (self.returnSequences) :
      output = self.hiddenPrev[1:]
    else: 
      output = self.hiddenPrev[-1]
    return output

  def getData(self):
    return [
      {
        'type': 'lstm',
        'params': {
          'inputSize': self.inputSize,
          'nCells': self.nCells,
          'returnSequences': self.returnSequences,
          'W_i': self.inputParam.w.tolist(),
          'W_f': self.forgetParam.w.tolist(),
          'W_c': self.cellParam.w.tolist(),
          'W_o': self.outputParam.w.tolist(),
          'U_i': self.inputParam.u.tolist(),
          'U_f': self.forgetParam.u.tolist(),
          'U_f': self.forgetParam.u.tolist(),
          'U_c': self.cellParam.u.tolist(),
          'U_o': self.outputParam.u.tolist(),
          'b_i': self.inputParam.b.tolist(),
          'b_f': self.forgetParam.b.tolist(),
          'b_c': self.cellParam.b.tolist(),
          'b_o': self.outputParam.b.tolist()
        }
      }
    ]
  
  def getName(self):
    return 'LSTM'
  
  def getShapeOutput(self):
    return (None, self.nCells)
  
  def getParameterCount(self):
    return (self.nAttributes * self.nCells * 4) + (self.nCells * self.nCells * 4) + (self.nCells * 4)

### TESTING ###
if __name__ == "__main__":
  inputData = np.array([[0.5, 3], [1, 2]])
  print(inputData.shape)
  lstmLayer = LSTMLayer(inputSize=(2,2), nCells=64)
  output = lstmLayer.forward(inputData)
  print("===")
  print(f"output={output}")