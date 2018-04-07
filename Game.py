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
    
    def run_simulation(self, num_episodes):
        for i in range(num_episodes):
            self.run_episode()
    
    def run_episodes(self):
        self.old_obs = self.reshape(self.env.reset())
        
        while True:
            done = self.run_tick()
            if done:
                break

    def run_tick(self):
        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        return done

if __name__ == '__main__':
    game = Game()
    
