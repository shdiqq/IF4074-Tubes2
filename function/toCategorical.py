import numpy as np

def toCategorical(label, numClasses):
  if numClasses == 1:
    newLabel = [[label[i]] for i in range(len(label))]
  else :
    newLabel = np.zeros((len(label), numClasses), dtype=int)
    for i in range (len(label)):
      newLabel[i, label[i]] = 1
  return(newLabel)