import tensorflow as tf
import numpy as np

class ConvNet1():
    ''' 
        SETUP 
    '''
    def __init__(self, params):
        self.set_hyperparams(params)
        
        # array of past events (old_state, new_state, action, reward) 
        self.memory = []

    def set_hyperparams(self, params):
        self.learning_rate = params['learning_rate']
        self.learning_rate_decay = params['learning_rate_decay']
        self.learning_rate_min = params['learning_rate_min']
        
        self.exploration_rate =  params['exploratin_rate']
        self.exploratin_rate_decay = params['exploration_rate_decay']
        self.exploration_rate_min = params['exploratin_rate_min']

        self.discount_rate = params['discount_rate']

    def set_obs_dim(self, obs_dim):
        self.obs_dim = obs_dim

    def calc_num_filters(self):
        assert self.obs_dim
        return True
    
    '''
        MODEL
    '''
    def create_conv_layer(self, inputs, num_filters, kernal_size):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=num_filters),
            kernal_size=kernal_size,
            padding="same",
            activation=tf.nn.relu)

    def create_pool_layer(self, inputs, pool_size, num_strides):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=pool_size,
            strides=num_strides)

    def build_model(self):
        # -1 for dynamically sized batch
        input_layer = tf.reshape(features['x'], [-1, 224, 320, 3])
        
        conv1 = self.create_conv_layer(input_layer, 128, [5, 5])
        
        pool1 = self.create_pool_layer(conv1, [2, 2], 2)
        
        conv2 = self.create_pool_layer(pool1, 256, [5, 5])
        
        pool2 = self.create_pool_layer(conv2, [2, 2], 2)

        pool2_flat = tf.reshape(pool2, [-1, 56 * 80 * 256]
        
if __name__ == '__main__':    
    convnet = ConvNet1()
    convnet.set_hyperparams(0.0001)
        
