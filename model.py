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
import inputdata

lamda = 1e-6
# import input_data
# import mnist
# import fully_connected_feed as ff

def one_hot(targets,n_class = 10):
    axis = targets.ndim
    ret_val = (np.arange(n_class) == np.expand_dims(targets,axis = axis)).astype(int)
    return ret_val

def fill_feed_dict(imgs_data,lbls_data,imgs_pl,lbls_pl):
	
	feed_dict = {
		imgs_pl : imgs_data,
		lbls_pl : lbls_data
		#is_training : is_training_val
	}
	return feed_dict

def eval_once_test(sess,eval_correct,imgs_pl,lbls_pl,data_set):
	true_count = 0
	train_data = np.array(data_set['data'])
	train_labels = one_hot(np.array(data_set['labels']))
	for i in range(500):
		mini_test_imgs = train_data[i*20:(i+1)*20,:]
		mini_test_lbls = train_labels[i*20:(i+1)*20,:]
		feed_dict = fill_feed_dict(mini_test_imgs,mini_test_lbls,imgs_pl,lbls_pl)
		true_count += sess.run(eval_correct,feed_dict = feed_dict)
	true_count/=500
	#precision = float(true_count) / inputdata.num_test_examples
	#print('Testing Data Eval:')
	print(' Accuracy: %0.04f' %(true_count))

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/tmp/tensorflow/parallel/',"""Directory to write the event file and chkpoints""")
tf.app.flags.DEFINE_string('num_gpus', 2,"""Nummber of GPUs to use""")
tf.app.flags.DEFINE_string('log_device_placement', True, """Whether to log device placement""")
tf.app.flags.DEFINE_string('batch_size', 100," ")
tf.app.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', " ")
tf.app.flags.DEFINE_string('fake_data',False, " ")


images = tf.placeholder(tf.float32,[None,3072])
labels = tf.placeholder(tf.float32,[None,10])

with tf.device("/gpu:0"):
	w1_0 = tf.Variable(tf.truncated_normal([3072,1024],stddev = 0.01), name = 'task_0_w1')
	b1_0 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_0_b1')
	l2_w1_0 = tf.nn.l2_loss(w1_0)

	w2_0 = tf.Variable(tf.truncated_normal([2048,1024],stddev = 0.01), name = 'task_0_w2')
	b2_0 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_0_b2')
	l2_w2_0 = tf.nn.l2_loss(w2_0)	

	w3_0 = tf.Variable(tf.truncated_normal([2048,1024],stddev = 0.01), name = 'task_0_w3')
	b3_0 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_0_b3')
	l2_w3_0 = tf.nn.l2_loss(w3_0)

	w4_0 = tf.Variable(tf.truncated_normal([2048,1024],stddev = 0.01), name = 'task_0_w4')
	b4_0 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_0_b4')
	l2_w4_0 = tf.nn.l2_loss(w4_0)

	w5_0 = tf.Variable(tf.truncated_normal([2048,10],stddev = 0.01), name = 'task_0_w5')
	b5_0 = tf.Variable(tf.constant(0.1,shape = [10]),name = 'task_0_b5')
	l2_w5_0 = tf.nn.l2_loss(w5_0)
	l2_t0 = l2_w1_0 + l2_w2_0 + l2_w3_0 + l2_w4_0 + l2_w5_0
	

with tf.device("/gpu:1"):
	w1_1 = tf.Variable(tf.truncated_normal([3072,1024],stddev = 0.01), name = 'task_1_w1')
	b1_1 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_1_b1')
	l2_w1_1 = tf.nn.l2_loss(w1_1)

	w2_1 = tf.Variable(tf.truncated_normal([2048,1024],stddev = 0.01), name = 'task_1_w2')
	b2_1 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_1_b2')
	l2_w2_1 = tf.nn.l2_loss(w2_1)

	w3_1 = tf.Variable(tf.truncated_normal([2048,1024],stddev = 0.01), name = 'task_1_w3')
	b3_1 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_1_b3')
	l2_w3_1 = tf.nn.l2_loss(w3_1)

	w4_1 = tf.Variable(tf.truncated_normal([2048,1024],stddev = 0.01), name = 'task_1_w4')
	b4_1 = tf.Variable(tf.constant(0.1,shape = [1024]),name = 'task_1_b4')
	l2_w4_1 = tf.nn.l2_loss(w4_1)

	w5_1 = tf.Variable(tf.truncated_normal([2048,10],stddev = 0.01), name = 'task_1_w5')
	b5_1 = tf.Variable(tf.constant(0.1,shape = [10]),name = 'task_1_b5')
	l2_w5_1 = tf.nn.l2_loss(w5_1)
	l2_t1 = l2_w1_1 + l2_w2_1 + l2_w3_1 + l2_w4_1 + l2_w5_1

with tf.device("/cpu:0"):
	h1_task0 = tf.matmul(images,w1_0)+b1_0
	h1_task1 = tf.matmul(images,w1_1)+b1_1
	h1_stack = tf.concat(1,[h1_task0,h1_task1])
	h2_task0 = tf.matmul(h1_stack,w2_0)+b2_0
	h2_task1 = tf.matmul(h1_stack,w2_1)+b2_1
	h2_stack = tf.concat(1,[h2_task0,h2_task1])
	h3_task0 = tf.matmul(h2_stack,w3_0)+b3_0	
	h3_task1 = tf.matmul(h2_stack,w3_1)+b3_1
	h3_stack = tf.concat(1,[h3_task0,h3_task1])
	h4_task0 = tf.matmul(h3_stack,w4_0)+b4_0
	h4_task1 = tf.matmul(h3_stack,w4_1)+b4_1
	h4_stack = tf.concat(1,[h4_task0,h4_task1])
	h5_task1 = tf.matmul(h4_stack,w5_1) + b5_1
	h5_stack = h5_task1

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h5_stack, labels = labels, name = "cross_entropy_mean")) + lamda*(l2_t0 + l2_t1)
	global_step = tf.Variable(0)

	train_op = tf.train.AdamOptimizer(1e-4).minimize(loss,global_step = global_step)
	correct = tf.equal(tf.argmax(h5_stack,1), tf.argmax(labels,1), name = 'Correct_prediction')
	eval_correct = tf.reduce_mean(tf.cast(correct, tf.float32), name = 'Correct_prediction_mean')
	saver = tf.train.Saver()
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()

	data_sets = inputdata.batch_arr
	config = tf.ConfigProto(allow_soft_placement = True)
	config.gpu_options.allow_growth = True
	step = 0
	with tf.Session(config = config) as sess:
		sess.run(init_op)
		k=0
		while k<10:
			loss_value = 0
			for i in range(5):	# Iterating on the number of batches
				start_time = time.time()
				batch = data_sets[i]
				lbls_data = one_hot(np.array(batch['labels']))
				imgs_data = np.array(batch['data'])
				for j in range(25):	#Iterating on mini batches of a given batch
					mini_batch_imgs = imgs_data[j*400:(j+1)*400,:]
					mini_batch_lbls = lbls_data[j*400:(j+1)*400,:]
					feed_dict = fill_feed_dict(mini_batch_imgs,mini_batch_lbls,images,labels)
					_,loss_val,step = sess.run([train_op,loss,global_step], feed_dict = feed_dict)
					loss_value+=loss_val
				duration = time.time()-start_time
				print("batch %d in %.3f sec"%(i+1,duration))
			loss_value/=125
			eval_once_test(sess,eval_correct,images,labels,inputdata.test_batch)
			print("Epoch %d loss_value %.4f "%(k+1,loss_value))
			k+=1