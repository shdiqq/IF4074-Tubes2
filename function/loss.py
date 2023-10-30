import numpy as np

# LOSS FUNCTION
def sumSquareError(target, output, derivative=False):
	if(derivative):
		return (output - target)
	else:
		return ((target - output)**2) / 2

def crossEntropy(target, output, derivative=False):
	if(derivative):
		return -(target / output) + (1 - target) / (1 - output)
	else:
		return -(target * np.log(output))\

def meanSquareError(target, output):
    return np.mean((target - output) ** 2)

def rootMeanSquareError(target, output):
    return np.sqrt(meanSquareError(target, output))