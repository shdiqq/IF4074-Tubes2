def spatialSize(inputHeight, inputWidth, filterSize, padding, stride):
  outputHeight = ( inputHeight - filterSize + 2 * padding ) // stride + 1
  outputWidth = ( inputWidth - filterSize + 2 * padding ) // stride + 1
  return outputHeight, outputWidth