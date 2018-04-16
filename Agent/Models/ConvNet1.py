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

        self.exploration_rate =  params['exploration_rate']
        self.exploration_rate_decay = params['exploration_rate_decay']
        self.exploration_rate_min = params['exploration_rate_min']

        self.discount_rate = params['discount_rate']

    def set_obs_dim(self, obs_dim):
        self.obs_dim = obs_dim

    def set_action_dim(self, action_dim):
        self.action_dim = action_dim

    '''
        MODEL
    '''
    def create_conv_layer(self, inputs, num_filters, kernal_size):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=num_filters,
            kernal_size=kernal_size,
            padding="same",
            activation=tf.nn.relu)

    def create_pool_layer(self, inputs, pool_size, num_strides):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=pool_size,
            strides=num_strides)

    def create_dense_layer(self, inputs, units):
        return tf.layers.dense(
            inputs=inputs,
            units=units,
            activation=tf.nn.relu)

    def create_dropout_layer(self, inputs, rate):
        return tf.layers.dropout(
            inputs=inputs,
            rate=rate)

    def build_model(self):
        assert self.obs_dim
        assert self.action_dim
        # -1 for dynamically sized batch
        input_layer = tf.reshape(tf.float32, [None, 224, 320, 3])

        conv1 = self.create_conv_layer(input_layer, 32, [8, 8], [4, 4])

        pool1 = self.create_pool_layer(conv1, [2, 2], 2)

        conv2 = self.create_pool_layer(pool1, 256, [5, 5])

        pool2 = self.create_pool_layer(conv2, [2, 2], 2)

        pool2_flat = tf.reshape(pool2, [-1, 56 * 80 * 256])

        dense1 = self.create_dense_layer(pool2_flat, 1024)

        dropout1 = self.create_dropout_layer(dense1, 0.4)

        dense2 = self.create_dense_layer(dropout1, 512)

        dropout2 = self.create_dropout_layer(dense2, 0.2)

        output_layer = tf.layers.dense(dropout2, self.action_dim) # player actions

if __name__ == '__main__':
    convnet = ConvNet1()
    convnet.set_hyperparams(0.0001)
