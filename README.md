# Robotics_FCN_Project

Note: The "Project_Follow_me_writeup.pdf" contains more details and illustrative explanation of the project.

## Project Objective: 
The project objective is to build a segmentation network, train it, validate it, and deploy it in the Follow Me project. 
A drone will use the trained NN model to track and follow a single hero target using the new network pipeline.

Project Steps: 

Step:1 Data Collection

A starting dataset of 4131 images with mask for training and 1184 images with mask for validation was provided. An additional of 3679 images were collected using the QuadSim simulator (by Unity).

Step:2 Image Preprocessing 

Before the network is trained, the images undergoes a preprocessing step. The preprocessing step transforms the depth masks from the simulator, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset

Step:3 FCN Layers 

Convolutional Network are Neural Networks that share their parameters across space. A patch/kernel of specific depth is being convoluted to input image/layer pixels to generate each layer in this network. CNN learns to recognize basic lines and curves, then shapes and blobs, and then increasingly complex objects within the image. A Fully Convolutional Network(FCN) is a special type of convolutional network which preserve the spatial information throughout the image space by decoding or de-convoluting the encoded or convoluted network.

Step:4 Build the Model 

In the following cells, an FCN is built to train a model to detect and locate the hero target within an image. 
The steps are: 

• Create an encoder_block

• Create a decoder_block 

• Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s).

Step:4 Training 

Training steps are: 

• Define the Keras model and compile it for training 

• Data iterators for loading the training and validation data Note: For this project, the helper code in data_iterator.py will resize the copter images to 160x160x3 to speed up training.

Results:

• A Fully Convolutional Neural Network model was developed which consists of 3 hidden layers of the encoders, one 1x1 convolution layer and 3 hidden layers of the decoders.

• Segmentation technique: semantic

• Training and validation sets:

No. of training images and mask: 7810

No. of validation images and mask: 2504

• Optimization of the network model using following hyper-parameters (final values):

learning_rate = 0.009

batch_size = 128

num_epochs = 60

steps_per_epoch = 61

validation_steps = 20

workers = 8

• Accuracy measurement of the NN using Intersection over Union IoU metric (IoU): 0.42

##Future Enhancement:

The project still has scopes to improve the prediction and tune hyper parameters. Those parameters should further be tuned to obtain 0 false positives and 0 false negatives. Training model could be tuned further to optimized to ensure detecting target object while in patrol mode more accurately.

The training data is used to train the FCN model to identify human shape and find the difference between target(and) non-target. In order to follow another object (dog, cat, car, etc.) instead of a human the model needs to be trained with the pictures or training data of the different target object.
