import numpy as np

class FlattenLayer():
  def __init__(self):
    pass

  def forward(self, inputData):
    return np.ravel(inputData)

### TESTING ###
if __name__ == "__main__":
  matrix = np.array(
    [
      [
        [
          [1,11,2],
          [1,10,4],
          [6,12,8],
        ],
        [
          [7,1,2],
          [5,-1,2],
          [7,-4,2],
        ],
        [
          [-2,23,2],
          [2,20,4],
          [8,6,6],
        ]
      ]
    ]
  )
  print(matrix[0].shape)
  print("=====")
  flattenLayer = FlattenLayer()
  newMatrix = flattenLayer.forward(matrix[0])
  print(len(newMatrix))