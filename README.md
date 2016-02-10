# Two-Gesture Recognizer
Trained a gesture recognizer neural net using Tensorflow with 500 32x32 images of each gesture.
The network is a 3-layer network with 2 hidden layers (1024-50-10-2).
The network uses stochastic gradient descent to optimize the cross entropy cost frunction.
It also uses constant learning rate.
The training images were generated using OpenCV to capture 500 frames of each gesture. The gestures were varied by moving around the camera field of view. See folders /one and /two.
