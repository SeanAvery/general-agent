import retro

if __name__ == '__main__':
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        print('obs', obs)
        print('rew', rew)
        if done:
            break
