import numpy as np

# ACTIVATION FUNCTION

def sigmoid(net: float, derivative: bool = False) -> float:
	if (derivative) :
		return ( (1 / (1 + np.exp(-net)) ) * ( 1 - (1 / (1 + np.exp(-net))) ) )
	else :
		return ( 1 / (1 + np.exp(-net) ) )

def relu(net: float, derivative: bool = False) -> float:
	if (derivative) :
		if ( net  < 0 ) :
			return(0)
		else :
			return(1)
	else :
		return (np.maximum(0, net))