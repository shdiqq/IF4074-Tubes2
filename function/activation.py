import numpy as np

# ACTIVATION FUNCTION

def linear(net: np.ndarray, derivative: bool = False):
	if (derivative) :
		return (np.full((net.shape), 1))
	else :
		return (net)

def sigmoid(net: np.ndarray, derivative: bool = False):
	if (derivative) :
		s = sigmoid(net)
		return (s * (1 - s))
	else :
		net = np.clip(net, -500, 500)
		return ( 1 / (1 + np.exp(-net) ) )

def relu(net: np.ndarray, derivative: bool = False):
	if (derivative) :
		return (net > 0).astype(float)
	else :
		return (np.maximum(0, net))