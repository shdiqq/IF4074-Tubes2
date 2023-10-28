import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import sigmoid

class LSTMParameter:
  def __init__(self, inputSize, nCells, U, W, b):
    # self.u = np.random.rand(nCells, inputSize)
    # self.w = np.random.rand(inputSize, nCells)
    # self.b = np.random.rand(1, nCells) 
    self.u = U
    self.w = W
    self.b = b

class LSTMLayer():
  def __init__(self, inputSize, nCells, U, W, b):
    self.inputData = None
    self.inputSize = inputSize
    self.nCells = nCells

    self.cellPrev = np.zeros((nCells+1, 1))
    self.hiddenPrev = np.zeros((nCells+1, 1))

    # 4 gate param
    self.forgetParam = LSTMParameter(self.inputSize, self.nCells, U[0], W[0], b[0])
    self.inputParam = LSTMParameter(self.inputSize, self.nCells, U[1], W[1], b[1])
    self.cellParam = LSTMParameter(self.inputSize, self.nCells, U[2], W[2], b[2])
    self.outputParam = LSTMParameter(self.inputSize, self.nCells, U[3], W[3], b[3])

  def forgetGate(self, timestep):
    # sigmoid(inputData * Uf + hiddenPrev * Wf + Bf)
    net_f = np.dot(self.forgetParam.u, self.inputData[timestep]) + np.dot(self.forgetParam.w, self.hiddenPrev[timestep]) + self.forgetParam.b
    ft = sigmoid(net_f)
    print(f"ft={ft}")
    return (ft)

  def inputGate(self, timestep):
    # sigmoid(inputData * Ui + hiddenPrev * Wi + Bi)
    net_i = np.matmul(self.inputParam.u, self.inputData[timestep]) + np.dot(self.inputParam.w, self.hiddenPrev[timestep]) + self.inputParam.b
    it = sigmoid(net_i)
    print(f"it={it}")

    # tanH(inputData * Uc + hiddenPrev * Wc + Bc)
    net_candidate = np.dot(self.cellParam.u, self.inputData[timestep]) + np.dot(self.cellParam.w, self.hiddenPrev[timestep]) + self.cellParam.b
    candidate_t = np.tanh(net_candidate)
    print(f"candidate_t={candidate_t}")
    return it, candidate_t

  def cellState(self, timestep, ft, it, candidate_t):
    # candidate * cellPrev + inputGate * candidate
    ct = np.multiply(ft, self.cellPrev[timestep]) + np.multiply(it, candidate_t)
    return ct

  def outputGate(self, timestep, ct):
    # sigmoid(inputData * Uo + hiddenPrev * Wo + Bo)
    net_o = np.dot(self.outputParam.u, self.inputData[timestep]) + np.dot(self.outputParam.w, self.hiddenPrev[timestep]) + self.outputParam.b
    ot = sigmoid(net_o)
    print(f"ot={ot}")

    # output * tanH(cellState)
    ht = np.multiply(ot, np.tanh(ct))
    return ht
  
  def forward(self, inputData):
    self.inputData = inputData
    for i in range(self.nCells):
      ft = self.forgetGate(i)
      it, candidate_t = self.inputGate(i)
      ct = self.cellState(i, ft, it, candidate_t)
      ht = self.outputGate(i, ct)

      self.cellPrev[i+1] = ct
      print(f"ct = {self.cellPrev[i+1]}")
      self.hiddenPrev[i+1] = ht
      print(f"ht = {self.hiddenPrev[i+1]}")
    
    output = self.hiddenPrev[-1]
    return output

  def getData(self):
    return [
        {
            'type': 'lstm',
            'params':{
                
            }
        }
    ]

### TESTING ###
if __name__ == "__main__":
  inputData = np.array([[0.5, 3], [1, 2]])
  U = [[0.5, 0.75], [0.81, 0.2], [0.35, 0.45], [0.4, 0.6]]
  W = [0.3, 0.7, 0.35, 0.4]
  b = [0.4, 0.55, 0.25, 0.5]
  lstmLayer = LSTMLayer(inputSize=2, nCells=2, U=U, W=W, b=b)
  output = lstmLayer.forward(inputData)
  print("===")
  print(f"output={output}")