Uses the data collection file to constantly take pictures of the screen, resize and edit the images, and save them to a folder depending on which key is being pressed when 
the picture is taken. The image data is then loaded into tensorflow, a machine learning package from google, and used to train a convolutional neural network. The
key presses are used as the features in this network and given an image taken, it can accurately predict which key should be pressed in response. Once the model is 
trained, it is used in the main file. This is where images are taken of the screen, edited, and fed into the network, and the output is used to press a key, moving the car in the 
correct direction.
