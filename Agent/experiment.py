from Models.ConvNet1 import ConvNet1
from Game import Game

hyperparams1 = {
    "learning_rate": 1,
    "learning_rate_decay": 0.999,
    "learning_rate_min": 0.001,
    "exploration_rate": 1,
    "exploration_rate_decay": 0.999,
    "exploration_rate_min": 0.001,
    "discount_rate": 0.95
}


if __name__ == '__main__':
    convnet = ConvNet1(hyperparams1)
    game = Game()
    convnet.set_action_dim(game.action_dim)
    convnet.set_obs_dim(game.obs_dim)
