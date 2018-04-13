import retro
import numpy as np

class Game():
    def __init__(self):
        self.env = retro.make(game='Airstriker-Genesis', state='Level1')
        self.get_action_dim()
        self.get_obs_dim()
    
    '''
        SETUP
        action_size: [Int]
        obs_size: Numpy np.ndarray
        action_size: Array
    ''' 
    def get_action_dim(self):
        self.action_dim = int(self.env.action_space.n)

    def get_obs_dim(self):
        self.obs_dim = self.env.observation_space.low.shape
    
    '''
        UTILS
    '''

    def reshape(self, vector):
        return vector.reshape(1, self.obs_size) 

    '''
        SIMULATION
    '''
    def run_simulation(self, num_episodes):
        for i in range(num_episodes):
            self.run_episode()
    
    def run_episode(self):
        self.old_obs = self.env.reset()
        while True:
            done = self.run_tick()
            if done:
                break

    def run_tick(self):
        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        print("reward", reward) 
        return done

