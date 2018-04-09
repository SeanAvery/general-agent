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
    
    def create_conv_layer(self):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=32,
            kernal_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

    def build_model(self):
        input_layer = tf.reshape(features['x'], [-1, 224, 320, 3])
        
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=32,
            kernal_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        
        pool1 = tf.layers.makx_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        

if __name__ == '__main__':    
    convnet = ConvNet1()
    convnet.set_hyperparams(0.0001)
        
