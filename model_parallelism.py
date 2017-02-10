from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import fully_connected_feed as ff

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/tmp/tensorflow/parallel/',"""Directory to write the event file and chkpoints""")
tf.app.flags.DEFINE_string('num_gpus', 2,"""Nummber of GPUs to use""")
tf.app.flags.DEFINE_string('log_device_placement', True, """Whether to log device placement""")
tf.app.flags.DEFINE_string('batch_size', 64," ")
tf.app.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', " ")
tf.app.flags.DEFINE_string('fake_data',False, " ")

IMG_PIXELS = 784
NUM_CLASSES = 10

TOWER_NAME = 'tower_'

# def inference(images):
# 	#FC1
# 	w1 = tf.Variable(tf.truncated_normal([784,256],stddev = 0.01), name = 'w1')
# 	b1 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'b1')

# 	h1 = tf.matmul(images,w1)+b1

# 	#FC2
# 	w2 = tf.Variable(tf.truncated_normal([512,512],stddev = 0.01), name = 'w2')
# 	b2 = tf.Variable(tf.constant(0.1,shape = [512]),name = 'b2')

# 	h2 = tf.matmul(h1,w2)+b2

# 	#FC3
# 	w3 = tf.Variable(tf.truncated_normal([1024,256],stddev = 0.01), name = 'w3')
# 	b3 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'b3')

# 	h3 = tf.matmul(h2,w3)+b3

# 	return h3
cluster = tf.train.ClusterSpec({"local" : ["localhost:2222"],"ps" : ["localhost:2223", "localhost:2224"]})

images = tf.placeholder(tf.float32,[None,784])
labels = tf.placeholder(tf.float32,[None,10])
#x = tf.constant(100)

with tf.device("/job:ps/task:1"):
	w1_0 = tf.Variable(tf.truncated_normal([784,256],stddev = 0.01), name = 'task_0_w1')
	b1_0 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_0_b1')

	#h1_task0 = tf.matmul(images,w1_0)+b1_0
	# h1_stack_0 = np.stack(h1_task0,h1_task1)

	w2_0 = tf.Variable(tf.truncated_normal([512,512],stddev = 0.01), name = 'task_0_w2')
	b2_0 = tf.Variable(tf.constant(0.1,shape = [512]),name = 'task_0_b2')

	# h2_task0 = tf.matmul(h1_stack,w2_0)+b2_0
	# h2_stack_0 = np.stack(h2_task0,h2_task1)

	w3_0 = tf.Variable(tf.truncated_normal([1024,10],stddev = 0.01), name = 'task_0_w3')
	b3_0 = tf.Variable(tf.constant(0.1,shape = [10]),name = 'task_0_b3')

	# h3_task0 = tf.matmul(h2_stack,w3_0)+b3_0
	#h3_stack_0 = np.stack(h3_task0,h3_task1)


with tf.device("/job:ps/task:2"):
	w1_1 = tf.Variable(tf.truncated_normal([784,256],stddev = 0.01), name = 'task_1_w1')
	b1_1 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_1_b1')

	# h1_task1 = tf.matmul(images,w1_1)+b1_1
	# h1_stack_1 = np.stack(h1_task0,h1_task1)

	w2_1 = tf.Variable(tf.truncated_normal([512,512],stddev = 0.01), name = 'task_1_w2')
	b2_1 = tf.Variable(tf.constant(0.1,shape = [512]),name = 'task_1_b2')

	# h2_task1 = tf.matmul(h1_stack,w2_1)+b2_1
	# h2_stack_1 = np.stack(h2_task0,h2_task1)

	w3_1 = tf.Variable(tf.truncated_normal([1024,10],stddev = 0.01), name = 'task_1_w3')
	b3_1 = tf.Variable(tf.constant(0.1,shape = [10]),name = 'task_1_b3')

	# h3_task1 = tf.matmul(h2_stack,w3_1)+b3_1

with tf.device("/job:local/task:0"):
	h1_task0 = tf.matmul(images,w1_0)+b1_0
	h1_task1 = tf.matmul(images,w1_1)+b1_1
	h1_stack = tf.concat([h1_task0,h1_task1],0)
	h2_task0 = tf.matmul(h1_stack,w2_0)+b2_0
	h2_task1 = tf.matmul(h1_stack,w2_1)+b2_1
	h2_stack = tf.concat([h2_task0,h2_task1],0)
	h3_task0 = tf.matmul(h2_stack,w3_0)+b3_0
	h3_task1 = tf.matmul(h2_stack,w3_1)+b3_1
	h3_stack = tf.concat([h3_task0,h3_task1],0)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h3_stack, labels = labels, name = "cross_entropy_mean"))
	global_step = tf.Variable(0)

	train_op = tf.train.AdamOptimizer(0.01).minimize(loss,global_step = global_step)
	saver = tf.train.Saver()
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()


sv = tf.train.Supervisor(is_chief = 0,
						logdir = "/tmp/train_logs",
						init_op = init_op,
						summary_op = summary_op,
						saver = saver,
						global_step = global_step,
						save_model_secs = 600)

data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
# with tf.Session("grpc://localhost:2222",config = config) as sess:
# 	step = 0
# 	sess.run(init_op)
# 	while step < 10:
# 		feed_dict = ff.fill_feed_dict(data_sets, images, labels)
# 		_,step = sess.run([train_op,global_step],feed_dict = feed_dict)

with sv.managed_session(server.target, config= config) as sess:
	step = 0 
	while not sv.should_stop() and step<10:
		feed_dict = ff.fill_feed_dict(data_sets, images, labels)
		_,step = sess.run([train_op,global_step], feed_dict = feed_dict)

sv.stop()