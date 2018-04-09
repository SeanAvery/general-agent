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


''' Tests '''
if __name__ == '__main__':
    
    convnet = ConvNet1()
    convnet.set_hyperparams(0.0001)
        
