import numpy as np
from keras import backend as K
import tensorflow as tf

def bilinear_interpolator(shape):

    filter_size = shape[0]
    num_channels = 1

    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                (1 - abs(y - center) / scale_factor)
 
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to constant_initalizer and pass to get_variable
    bilinear_init = tf.constant_initializer(value=weights, dtype=tf.float32)

    return bilinear_init