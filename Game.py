import retro

class Game():
    def __init__(self):
        self.env = retro.make(game='Airstriker-Genesis', state='Level1')
        self.get_action_dim()
        self.get_obs_dim()

    def get_action_dim(self):
        self.action_size = self.env.action_space.n
        print('self.action_size', self.action_size)

    def get_obs_dim(self):
        self.obs_size = len(self.env.observation_space.low)
        print('self.obs_size', self.obs_size)

    def reshape(self, vector):
        return vector.reshape(1, self.obs_size) 
        

if __name__ == '__main__':
    game = Game()
    
