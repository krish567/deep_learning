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
from tensorflow.python.client import timeline
import inputdata

# import input_data
# import mnist
# import fully_connected_feed as ff

lamda = 1e-6

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

def conv2d(x, W,name):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)

def max_pool_3x3(x,name):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name = name)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', '/tmp/tensorflow/parallel/',"""Directory to write the event file and chkpoints""")
tf.app.flags.DEFINE_string('num_gpus', 2,"""Nummber of GPUs to use""")
tf.app.flags.DEFINE_string('log_device_placement', True, """Whether to log device placement""")
tf.app.flags.DEFINE_string('batch_size', 100," ")
tf.app.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data', " ")
tf.app.flags.DEFINE_string('fake_data',False, " ")

with tf.device("/cpu:0"):
	images = tf.placeholder(tf.float32,[None,3072], name = "images_pl")
	labels = tf.placeholder(tf.float32,[None,10], name = "labels_pl")

with tf.device("/gpu:0"):
	w1_0 = tf.Variable(tf.truncated_normal([5,5,3,512],stddev = 0.01), name = 'task_0_w1')
	b1_0 = tf.Variable(tf.constant(0.1,shape = [512]),name = 'task_0_b1')
	
	w2_0 = tf.Variable(tf.truncated_normal([5,5,1024,256],stddev = 0.01), name = 'task_0_w2')
	b2_0 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_0_b2')
	

	w3_0 = tf.Variable(tf.truncated_normal([5,5,512,256],stddev = 0.01), name = 'task_0_w3')
	b3_0 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_0_b3')

	w4_0 = tf.Variable(tf.truncated_normal([5,5,512,256],stddev = 0.01), name = 'task_0_w4')
	b4_0 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_0_b4')

	w5_0 = tf.Variable(tf.truncated_normal([16*16*512,10],stddev = 0.01), name = 'task_0_w5')
	b5_0 = tf.Variable(tf.constant(0.1,shape = [10]),name = 'task_0_b5')

	l2_w1_0 = tf.nn.l2_loss(w1_0, name = "l2_loss_w1_0")
	l2_w2_0 = tf.nn.l2_loss(w2_0, name = "l2_loss_w2_0")
	l2_w3_0 = tf.nn.l2_loss(w3_0, name = "l2_loss_w3_0")
	l2_w4_0 = tf.nn.l2_loss(w4_0, name = "l2_loss_w4_0")
	l2_w5_0 = tf.nn.l2_loss(w5_0, name = "l2_loss_w5_0")

	l2_t0 = l2_w1_0 + l2_w2_0 + l2_w3_0 + l2_w4_0 + l2_w5_0

with tf.device("/gpu:1"):
	w1_1 = tf.Variable(tf.truncated_normal([5,5,3,512],stddev = 0.01), name = 'task_1_w1')
	b1_1 = tf.Variable(tf.constant(0.1,shape = [512]),name = 'task_1_b1')
	
	w2_1 = tf.Variable(tf.truncated_normal([5,5,1024,256],stddev = 0.01), name = 'task_1_w2')
	b2_1 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_1_b2')
	

	w3_1 = tf.Variable(tf.truncated_normal([5,5,512,256],stddev = 0.01), name = 'task_1_w3')
	b3_1 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_1_b3')

	w4_1 = tf.Variable(tf.truncated_normal([5,5,512,256],stddev = 0.01), name = 'task_1_w4')
	b4_1 = tf.Variable(tf.constant(0.1,shape = [256]),name = 'task_1_b4')

	l2_w1_1 = tf.nn.l2_loss(w1_1, name = "l2_loss_w1_1")
	l2_w2_1 = tf.nn.l2_loss(w2_1, name = "l2_loss_w2_1")
	l2_w3_1 = tf.nn.l2_loss(w3_1, name = "l2_loss_w3_1")
	l2_w4_1 = tf.nn.l2_loss(w4_1, name = "l2_loss_w4_1")
	l2_w5_1 = tf.nn.l2_loss(w5_1, name = "l2_loss_w5_1")

	l2_t1 = l2_w1_0 + l2_w1_1 + l2_w2_0 + l2_w2_1 + l2_w3_0 + l2_w3_1 + l2_w4_0 + l2_w4_1 +l2_w5_0 + l2_w5_1

with tf.device("/gpu:0"):
	x_image_0 = tf.reshape(images, [-1,32,32,3])
	h1_task0 = tf.nn.relu(conv2d(x_image_0,w1_0,"h_conv1_0")+b1_0, name = "h_relu1_0")
with tf.device("/gpu:1"):
	x_image_1 = tf.reshape(images, [-1,32,32,3])
	h1_task1 = tf.nn.relu(conv2d(x_image_1,w1_1,"h_conv1_1")+b1_1, name = "h_relu1_1")
with tf.device("/cpu:0"):
	h1_stack = tf.concat(3,[h1_task0,h1_task1], name = "stack_l1")

with tf.device("/gpu:0"):
	h2_task0 = tf.nn.relu(conv2d(h1_stack,w2_0,"h_conv2_0")+b2_0, name = "h_relu2_0")
with tf.device("/gpu:1"):
	h2_task1 = tf.nn.relu(conv2d(h1_stack,w2_1,"h_conv2_1")+b2_1, name = "h_relu2_1")
with tf.device("/cpu:0"):
	h2_stack = tf.concat(3,[h2_task0,h2_task1], name = "stack_l2")

with tf.device("/gpu:0"):
	h3_task0 = max_pool_3x3(tf.nn.relu(conv2d(h2_stack,w3_0,"h_conv3_0")+b3_0, name = "h_relu3_0"),"pool1_0")
with tf.device("/gpu:1"):
	h3_task1 = max_pool_3x3(tf.nn.relu(conv2d(h2_stack,w3_1,"h_conv3_1")+b3_1, name = "h_relu3_1"),"pool1_1")
with tf.device("/cpu:0"):
	h3_stack = tf.concat(3,[h3_task0,h3_task1], name = "stack_l3")

with tf.device("/gpu:0"):
	h4_task0 = tf.nn.relu(conv2d(h3_stack,w4_0,"h_conv4_0")+b4_0, name = "h_relu4_0")
with tf.device("/gpu:1"):
	h4_task1 = tf.nn.relu(conv2d(h3_stack,w4_1,"h_conv4_1")+b4_1, name = "h_relu4_1")
with tf.device("/cpu:0"):
	h4_stack = tf.concat(3,[h4_task0,h4_task1], name = "stack_l4")

# with tf.device("/gpu:0"):
# 	h4_stack_reshape0 = tf.reshape(h4_stack,[-1,16*16*512])
# 	h5_task0 = tf.matmul(h4_stack_reshape0,w5_0, name = "matmul_h5_task0")+b5_0
with tf.device("/gpu:0"):
	h4_stack_reshape1 = tf.reshape(h4_stack,[-1,16*16*512])
	h5_task0 = tf.matmul(h4_stack_reshape1,w5_0, name = "matmul_h5_task1")+b5_0
with tf.device("/cpu:0"):
	h5_stack = h5_task1

with tf.device("/gpu:1"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = h5_stack, labels = labels, name = "cross_entropy_mean")) + lamda *(l2_t0 + l2_t1)
	opt = tf.train.AdamOptimizer(1e-5)

with tf.device("/gpu:1"):
	grads = opt.compute_gradients(loss)

with tf.device("/cpu:0"):	
	global_step = tf.Variable(0)
	apply_op = opt.apply_gradients(grads, global_step = global_step)

with tf.device("/gpu:0"):
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
	run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	k=0
	while k<3:
		loss_value = 0
		true_count = 0
		start_time_epoch = time.time()
		for i in range(5):	# Iterating on the number of batches
			start_time = time.time()
			batch = data_sets[i]
			lbls_data = one_hot(np.array(batch['labels']))
			imgs_data = np.array(batch['data'])
			for j in range(50):	#Iterating on mini batches of a given batch
				mini_batch_imgs = imgs_data[j*200:(j+1)*200,:]
				mini_batch_lbls = lbls_data[j*200:(j+1)*200,:]
				feed_dict = fill_feed_dict(mini_batch_imgs,mini_batch_lbls,images,labels)
				_,loss_val,step,count = sess.run([apply_op,loss,global_step,eval_correct], feed_dict = feed_dict, options = run_options, run_metadata = run_metadata)
				loss_value+=loss_val
				true_count+= count
			duration = time.time()-start_time
			print("batch %d in %.3f"%(i+1,duration))
			t1 = timeline.Timeline(run_metadata.step_stats)
			ctf = t1.generate_chrome_trace_format()
			with open('timeline.json','w') as f:
				f.write(ctf)
		loss_value/=250
		true_count/=250.0
		duration_epoch = time.time()-start_time_epoch
		print("Epoch %d accuracy %.4f loss_value %.4f in %.3f"%(k+1,true_count,loss_value,duration_epoch))
		k+=1
	eval_once_test(sess,eval_correct,images,labels,inputdata.test_batch)	