import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('num_gpus', 2,"""Number of gpus""")


def ___Weight_Bias(W_shape, b_shape):
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1/np.prod(W_shape)), name='weights')
    b = tf.Variable(tf.zeros(b_shape), name='biases')
    tf.summary.image('w',W[:,:,:,0:1],max_outputs = 5,
        collections = 'per_epoch')
    return W, b

def Flatten(x,len):
    return tf.reshape(x,[-1,len])

def Dense(x, filter_shape):
    """
    Args:
    x(tensor)
    input tensor

    filter_shape:
    filter_shape should be of the form: (num_input_neurons,num_output_neurons)

    Returns:
    A 'Tensor' that has the same type as input
    """
    W_shape = filter_shape
    b_shape = [filter_shape[1]]
    W, b = ___Weight_Bias(W_shape, b_shape)
    ret_val =  tf.matmul(x, W) + b
    return ret_val

def Conv2D(x, filter_shape, stride = 1, padding = 'VALID'):
    """
    Args:
    x(tensor): 
    input tensor
    
    filter_shape(list/tuple):
    filter shape should be of the form : (kernel_size_x,kernel_size_y,num_in_channels,num_out_channels)
    
    stride(int/list/tuple): 
    Can be an integer in which case stride_x = stride_y = stride will be assumed
    Can be a list/tuple where stride_x = stride[0] and stride_y = stride[1]
    
    padding(int/string):
    Default value: 'VALID'
    If padding is a number, a border of 'padding' pixels is added to the image before performing convolution.
    If padding is a string, it can be either 'SAME' or 'VALID'
    Output image is of same dimensions if padding is 'SAME'. No padding is done if padding is 'VALID'
    
    Returns:
    A 'Tensor' that has the same type as input
    
    Example Usage: 
    inputs = Inputs(None,128,128,1)
    L1 = conv2D(inputs,(3,3,1,32)
    """

    if isinstance(stride,int):
        strides = [1,stride,stride,1]
    if isinstance(stride,(list,tuple)):
        strides = [1,stride[0],stride[1],1]
        
    if isinstance(padding,int):
        tf.pad(x,[[0,0],[padding,padding],[padding,padding],[0,0]])
        padding = 'VALID'  
    
    W_shape = filter_shape
    b_shape = [filter_shape[3]]
    W, b = ___Weight_Bias(W_shape, b_shape)
    conv_out = tf.nn.conv2d(x, W, strides, padding)
    ret_val = conv_out + b
    # tf.summary.image('conv',ret_val[:,:,:,0:1],max_outputs = 5)
    return ret_val

def Relu(x):
    return tf.nn.relu(x)
    
def ConvRelu(x,shape,strides = 1,padding = 'VALID'):
    """
    Returns Relu(Conv2D(x,shape,strides,padding))
    """
    return Relu(Conv2D(x,shape,strides,padding))

def Tanh(x):
    return tf.nn.tanh(x)

def MaxPool2(x):
    """
    Does a 2x2 max-pooling to the input tensor
    Args:
    x(tensor): Input tensor
    Returns:
    A tensor of the same type as the input
    """
    ret_val = tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')
    return ret_val

def MaxPoolN(x,n):
    """
    Does max-pooling with a nxn kernel
    Args:
    x(tensor): input tensor
    n : size of the kernel
    Returns:
    A tensor of the same type as the input
    """
    ret_val = tf.nn.max_pool(x,ksize = [1,n,n,1],
        strides = [1,n,n,1])
    return ret_val

def Inputs(*args):
    """
    Args:
    d0,d1,d2,d3,....

    Returns: 
    A symbolic variable of the shape d0xd1xd2xd3xd4x...
    Note : Tensorflow's default ordering for image operations is BxHxWxC
    """
    if len(args) == 4:
        if args[1] != args[2]:
            print('\n WARNING : Tensorflow follows \'BHWC\' data format.You may have to change your input dimensions \n')
    inputs = tf.placeholder(tf.float32,args,name = 'Inputs')
    # tf.summary.image('inputs',inputs,10)
    return inputs

def OneHot(targets,num_class):
    """
    Args:
    x
    num_class : scalar representing number of target classes

    Returns:
    Applies a one-hot transformation to the targets
    """
    return tf.one_hot(targets,num_class,1,0)

def Targets(*args):
    """
    Args:
    d0,d1,d2,d3,...

    Returns:
    A symbolic variable for targets.
    """
    ret_val = tf.placeholder(tf.uint8,args,name = 'Targets')

    summary_tensor = tf.cast(tf.argmax(ret_val,axis = 3),tf.uint8)
    summary_tensor = summary_tensor[:,:,:,None]
    tf.summary.image('targets',summary_tensor,max_outputs = 5)
    return ret_val

def SpatialSoftmax2Cls(logit_map):
    exp_map = tf.exp(logit_map)
    evidence = tf.add(exp_map,tf.reverse(exp_map,[False,False,False,True]))
    ret_val = tf.div(exp_map,evidence,name = 'spatial_softmax')
    # summary_tensor = tf.argmax(ret_val,axis = 3)
    # tf.summary.image('predictions',summary_tensor,max_outputs = 15)
    return ret_val

def SpatialSoftmax(logit_map):
    exp_map = tf.exp(logit_map)
    sum_exp = tf.reduce_sum(exp_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp,
     tf.pack([1, 1, 1, tf.shape(logit_map)[3]]))
    ret_val = tf.div(exp_map,tensor_sum_exp,name = 'spatial_softmax')
    # summary_tensor = tf.argmax(ret_val,axis = 3)
    # tf.summary.image('predictions',summary_tensor,max_outputs = 15)
    return ret_val

#####################################################################

def Softmax(logits):
    return tf.nn.softmax(logits,name = 'softmax')

def DeconvRelu(x,shape,padding = 'VALID',stride = 2):
    """Spatial Full Convolution + Upsampling
    Args:
    x(tensor): Input tensor
    shape(tuple/list): 
    Returns:

    """
    W_shape = shape
    b_shape = [shape[2]]
    W,b = ___Weight_Bias(W_shape,b_shape)
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1]*2, x_shape[2]*2, shape[2]])
    strides = [1, stride, stride, 1]

    ret_val = Relu(tf.nn.conv2d_transpose(x, W, output_shape, 
        strides, padding) + b)
    # tf.summary.image('Deconv',ret_val[:,:,:,0:1],max_outputs = 5)
    return ret_val


def CropAndConcat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat(3, [x1_crop, x2])

"""
def BatchNorm(x):
    mean,variance = tf.nn.moments(x,axis)    

    ret_val = tf.nn.batch_normalization(x,mean,variance,
    beta,gamma,BN_EPSILON)
    #tf.image.summary(ret_val,)
    return ret_val

""" 