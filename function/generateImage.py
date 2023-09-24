from function.imageLabel import *
import numpy as np
import glob
import imageio.v2 as imageio

def generateImage() :
	ListImageInput = []
	ListClassLabel = []

	fileM_bear = glob.glob("data/train/bears/*.jpeg")
	fileM_panda = glob.glob("data/train/pandas/*.jpeg")

	for fileB in fileM_bear:
		input_imageB = imageio.imread(fileB)
		imageB_tensor = np.array(input_imageB)
		ListImageInput.append(imageB_tensor)
		ListClassLabel.append(imageLabel(fileB))

	for fileP in fileM_panda:
		input_imageP = imageio.imread(fileP)
		imageP_tensor = np.array(input_imageP)
		ListImageInput.append(imageP_tensor)
		ListClassLabel.append(imageLabel(fileP))

	listImageMatrix = np.array(ListImageInput, dtype="object")

	return (listImageMatrix, ListClassLabel)