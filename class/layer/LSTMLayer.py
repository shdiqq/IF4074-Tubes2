import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import sigmoid

class LSTMParameter:
  def __init__(self, inputSize):
    self.u = np.random.rand(inputSize)
    self.w = np.random.rand(inputSize)
    self.b = np.random.rand(inputSize)

class LSTMLayer():
  def __init__(self, inputSize, nCells, U=None, W=None, b=None):
    self.inputData = None
    self.inputSize = inputSize
    self.nCells = nCells

    print("Data di LSTM Layer")
    print(self.inputData)
    print(self.inputSize)
    self.cellPrev = np.zeros((nCells+1, inputSize))
    print(self.cellPrev)
    self.hiddenPrev = np.zeros((nCells+1, inputSize))

    # 4 gate param
    self.forgetParam = LSTMParameter(self.inputSize)
    print("Forget Param:", self.forgetParam.u, self.forgetParam.w, self.forgetParam.b)
    self.inputParam = LSTMParameter(self.inputSize)
    self.cellParam = LSTMParameter(self.inputSize)
    self.outputParam = LSTMParameter(self.inputSize)

  def forgetGate(self, timestep):
    # sigmoid(inputData * Uf + hiddenPrev * Wf + Bf)
    print("Forget Param:", self.forgetParam.u, self.forgetParam.w, self.forgetParam.b)
    print("Input Data:", self.inputData[timestep])
    print("len Input Data:", len(self.inputData[timestep]))
    print("Hidden Prev:", self.hiddenPrev[timestep])
    net_f = np.dot(self.forgetParam.u, self.inputData[timestep]) + np.dot(self.forgetParam.w, self.hiddenPrev[timestep]) + self.forgetParam.b
    ft = sigmoid(net_f)
    #print(f"ft={ft}")
    return (ft)

  def inputGate(self, timestep):
    # sigmoid(inputData * Ui + hiddenPrev * Wi + Bi)
    net_i = np.matmul(self.inputParam.u, self.inputData[timestep]) + np.dot(self.inputParam.w, self.hiddenPrev[timestep]) + self.inputParam.b
    it = sigmoid(net_i)
    #print(f"it={it}")

    # tanH(inputData * Uc + hiddenPrev * Wc + Bc)
    net_candidate = np.dot(self.cellParam.u, self.inputData[timestep]) + np.dot(self.cellParam.w, self.hiddenPrev[timestep]) + self.cellParam.b
    candidate_t = np.tanh(net_candidate)
    #print(f"candidate_t={candidate_t}")
    return it, candidate_t

  def cellState(self, timestep, ft, it, candidate_t):
    # candidate * cellPrev + inputGate * candidate
    ct = np.multiply(ft, self.cellPrev[timestep]) + np.multiply(it, candidate_t)
    return ct

  def outputGate(self, timestep, ct):
    # sigmoid(inputData * Uo + hiddenPrev * Wo + Bo)
    net_o = np.dot(self.outputParam.u, self.inputData[timestep]) + np.dot(self.outputParam.w, self.hiddenPrev[timestep]) + self.outputParam.b
    ot = sigmoid(net_o)
    #print(f"ot={ot}")

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

      print(ct)
      self.cellPrev[i+1] = ct
    #   print(self.cellPrev[i+1])
    #   print(f"ct = {self.cellPrev[i+1]}")
      self.hiddenPrev[i+1] = ht
      #print(f"ht = {self.hiddenPrev[i+1]}")
    
    output = self.hiddenPrev[-1]
    return output

  def getData(self):
    return [
      {
        'type': 'lstm',
        'params': {
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

### TESTING ###
if __name__ == "__main__":
  inputData = np.array([[0.5, 3], [1, 2]])
  lstmLayer = LSTMLayer(inputSize=2, nCells=2)
  output = lstmLayer.forward(inputData)
  print("===")
  print(f"output={output}")