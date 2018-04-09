import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.optimizers import Adam

class ConvNet():
    def build_model(self, obs_size, action_size):
        model = Sequential()
        
