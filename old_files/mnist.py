# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# def inference(images, hidden1_units, hidden2_units):
#   """Build the MNIST model up to where it may be used for inference.

#   Args:
#     images: Images placeholder, from inputs().
#     hidden1_units: Size of the first hidden layer.
#     hidden2_units: Size of the second hidden layer.

#   Returns:
#     softmax_linear: Output tensor with the computed logits.
#   """

#   hidden3_units = FLAGS.hidden3
#   hidden4_units = FLAGS.hidden4
#   hidden5_units = FLAGS.hidden5
#   hidden6_units = FLAGS.hidden6
#   # Hidden 1
#   with tf.name_scope('hidden1'):
#     weights = tf.Variable(
#         tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
#                             stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([hidden1_units]),
#                          name='biases')
#     hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
#   # Hidden 2
#   with tf.name_scope('hidden2'):
#     weights = tf.Variable(
#         tf.truncated_normal([hidden1_units, hidden2_units],
#                             stddev=1.0 / math.sqrt(float(hidden1_units))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([hidden2_units]),
#                          name='biases')
#     hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  
#   # Hidden 3
#   with tf.name_scope('hidden3'):
#     weights = tf.Variable(
#         tf.truncated_normal([hidden2_units, hidden3_units],
#                             stddev=1.0 / math.sqrt(float(hidden2_units))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([hidden3_units]),
#                          name='biases')
#     hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

#   # Hidden 4
#   with tf.name_scope('hidden4'):
#     weights = tf.Variable(
#         tf.truncated_normal([hidden3_units, hidden4_units],
#                             stddev=1.0 / math.sqrt(float(hidden3_units))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([hidden4_units]),
#                          name='biases')
#     hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)

#   # Hidden 5
#   with tf.name_scope('hidden5'):
#     weights = tf.Variable(
#         tf.truncated_normal([hidden4_units, hidden5_units],
#                             stddev=1.0 / math.sqrt(float(hidden4_units))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([hidden5_units]),
#                          name='biases')
#     hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)

#   # Hidden 6
#   with tf.name_scope('hidden6'):
#     weights = tf.Variable(
#         tf.truncated_normal([hidden5_units, hidden6_units],
#                             stddev=1.0 / math.sqrt(float(hidden5_units))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([hidden6_units]),
#                          name='biases')
#     hidden6 = tf.nn.relu(tf.matmul(hidden5, weights) + biases)

#   #Dropout
#   with tf.name_scope('dropout'):
#     keep_prob = tf.placeholder(tf.float32)
#     h_fc1_drop = tf.nn.dropout(hidden6, FLAGS.keep_prob)

#   # Linear
#   with tf.name_scope('softmax_linear'):
#     weights = tf.Variable(
#         tf.truncated_normal([hidden6_units, NUM_CLASSES],
#                             stddev=1.0 / math.sqrt(float(hidden6_units))),
#         name='weights')
#     biases = tf.Variable(tf.zeros([NUM_CLASSES]),
#                          name='biases')
#     logits = tf.matmul(h_fc1_drop, weights) + biases
#   return logits

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  # hidden3_units = FLAGS.hidden3
  # hidden4_units = FLAGS.hidden4
  # hidden5_units = FLAGS.hidden5
  # hidden6_units = FLAGS.hidden6
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
  #Convoluitonal layer 1
  with tf.name_scope('Conv 1 -> Relu 1'):
    weights = weight_variable([3,3,1,128])
    biases = bias_variable([128])

    x_image = tf.reshape(images,[-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image,weights)+biases)

  #Convolutional Layer 2
  with tf.name_scope('Conv 2 -> Relu2 -> pool1'):
    weights = weight_variable(3,3,128,128)
    biases = bias_variable(128)

    h_conv2 = tf.nn.relu(conv2d(h_conv1,weights)+biases)
    h_pool1 = max_pool_2x2(h_conv2)

  with tf.name_scope('Conv3 -> Relu3'):
    weights = weight_variable(3,3,128,256)
    biases = bias_variable(256)

    h_conv3 = tf.nn.relu(conv2d(h_pool1,weights)+biases)

  with tf.name_scope('Conv 4 -> Relu 4 -> Pool2'):
    weights = weight_variable(3,3,256,256)
    biases = bias_variable(256)

    h_conv4 = tf.nn.relu(conv2d(h_conv3,weights)+biases)
    h_pool2 = max_pool_2x2(h_conv4)

  #Fully Connected layer
  with tf.name_scope('Fully Connected Layer'):
    weights = weight_variable([7*7*256,2048])
    biases = bias_variable(2048)

    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,weights)+biases)

  #Dropout layer
  with tf.name_scope('dropout'):
    keep_prob = FLAGS.keep_prob
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([2048, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(7*7*256))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(h_fc1_drop, weights) + biases
  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('random',5)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(1e-5)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
