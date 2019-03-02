import os
import pickle
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import matplotlib.pyplot as plt
from tqdm import tqdm

%matplotlib inline

print('All modules imported.')

# Load the data
pickle_file = 'notMNIST_random_sample_150000.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_features']
  train_labels = pickle_data['train_labels']
  test_features = pickle_data['test_features']
  test_labels = pickle_data['test_labels']

  # Set flags for feature engineering.  This will prevent you from skipping an important step.
  is_features_normal = False
  is_labels_encod = False
  del pickle_data  # Free up memory

print('Data loaded.')
# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    a=0.1
    b=0.9
    image_data_max=255
    image_data_min=0
    norm_image_data=a+((image_data-image_data_min)*(b-a)/(image_data_max-image_data_min))
    return norm_image_data


### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255])),
    [0.1, 0.103137254902, 0.106274509804, 0.109411764706, 0.112549019608, 0.11568627451, 0.118823529412, 0.121960784314,
     0.125098039216, 0.128235294118, 0.13137254902, 0.9],
    decimal=3)
np.testing.assert_array_almost_equal(
    normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255])),
    [0.1, 0.103137254902, 0.13137254902, 0.162745098039, 0.194117647059, 0.225490196078, 0.830980392157, 0.865490196078,
     0.896862745098, 0.9])

if not is_features_normal:
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    is_features_normal = True

print('Tests Passed!')
if not is_labels_encod:
    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    # Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

print('Labels One-Hot Encoded')

assert is_features_normal, 'You skipped the step to normalize the features'
assert is_labels_encod, 'You skipped the step to One-Hot Encode the labels'

# Get randomized datasets for training and validation
train_features, valid_features, train_labels, valid_labels = train_test_split(
    train_features,
    train_labels,
    test_size=0.05,
    random_state=832289)

print('Training features and labels randomized and split.')

features_count = 784
labels_count = 10

# TODO: Set the hidden layer width. You can try different widths for different layers and experiment.
hidden_layer_width = 10

# TODO: Set the features, labels, and keep_prob tensors
features = tf.placeholder(tf.float32, [None, features_count])
labels = tf.placeholder(tf.float32, [None, labels_count])
keep_prob = 0.5


# TODO: Set the list of weights and biases tensors based on number of layers
weights = [
    tf.Variable([hidden_layer_width]),
    tf.Variable([labels_count])] 
#weights= weights.astype(np.float32)

print(weights)
biases = [
    tf.Variable(tf.zeros([hidden_layer_width])),
    tf.Variable(tf.zeros([labels_count]))]
print(biases)
#biases= biases.astype(np.float32)


### DON'T MODIFY ANYTHING BELOW ###
from tensorflow.python.ops.variables import Variable

assert features._op.name.startswith('Placeholder'), 'features must be a placeholder'
assert labels._op.name.startswith('Placeholder'), 'labels must be a placeholder'
assert all(isinstance(weight, Variable) for weight in weights), 'weights must be a TensorFlow variable'
assert all(isinstance(bias, Variable) for bias in biases), 'biases must be a TensorFlow variable'

assert features._shape == None or (\
    features._shape.dims[0].value is None and\
    features._shape.dims[1].value in [None, 784]), 'The shape of features is incorrect'
assert labels._shape  == None or (\
    labels._shape.dims[0].value is None and\
    labels._shape.dims[1].value in [None, 10]), 'The shape of labels is incorrect'

assert features._dtype == tf.float32, 'features must be type float32'
assert labels._dtype == tf.float32, 'labels must be type float32'

##Problem 3
#This problem would help you implement the hidden and output layers of your model. As it was covered in the classroom, you will need the following:
# TODO: Hidden Layers with ReLU Activation and dropouts. "features" would be the input to the first layer.
#hidden_layer_1 = None

hidden_layer_1 = tf.add(tf.matmul(features, weights[0]),biases[0])
hidden_layer_1 = tf.nn.relu(hidden_layer_1)

# TODO: Output layer
logits = tf.add(tf.matmul(hidden_layer_1, weights[1]), biases[1])

