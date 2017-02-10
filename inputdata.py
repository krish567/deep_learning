from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import cPickle
import tensorflow as tf

dir_add = "cifar-10-batches-py/"

def unpickle(file):
	fo = open(file,'rb')
	dictn = cPickle.load(fo)
	fo.close
	return dictn

batch1 = unpickle(dir_add+"data_batch_1")
batch2 = unpickle(dir_add+"data_batch_2")
batch3 = unpickle(dir_add+"data_batch_3")
batch4 = unpickle(dir_add+"data_batch_4")
batch5 = unpickle(dir_add+"data_batch_5")
test_batch = unpickle(dir_add+"test_batch")

batch_arr = np.array([batch1,batch2,batch3,batch4,batch5])

img_pxls = 3072

img_classes = 10

batch_size = 50
test_batch_size = 1000