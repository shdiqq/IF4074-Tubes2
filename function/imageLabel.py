def imageLabel(imageFileName) :  # Create a label for each image
  i_label = imageFileName.split("_", 1)[0]
  if 'bears' in i_label:  # Assign '0' for 'bear' images
    return 0
  elif 'pandas' in i_label:  # Assign '1' for 'panda' images
    return 1