import tensorflow as tf
import numpy as np

class ConvNet1():
    ''' 
        SETUP 
    '''
    def set_hyperparams(self, learning_rate):
        self.learning_rate = learning_rate

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
        
if __name__ == '__main__':    
    convnet = ConvNet1()
    convnet.set_hyperparams(0.0001)
        
