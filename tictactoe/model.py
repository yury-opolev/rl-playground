import random
import tensorflow as tf
from keras import layers
from keras import models
from keras import initializers

from game.game import Game
from game.agents.ai_agent import AIAgent

class Model(object):
    def __init__(self, model_path, summary_path, checkpoint_dir, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_dir = checkpoint_dir

        self.nn_model = models.Sequential([
            layers.Input(shape=(20,)),
            layers.Dense(20, activation='relu',
                         kernel_initializer=initializers.RandomNormal(stddev=0.1),
                         bias_initializer=initializers.RandomNormal(stddev=0.1)),
            layers.Dense(20, activation='relu',
                         kernel_initializer=initializers.RandomNormal(stddev=0.1),
                         bias_initializer=initializers.RandomNormal(stddev=0.1)),
            # result is a prediction of probability of winning: 1 - 'X' wins, 0 - 'X' looses
            layers.Dense(1, activation='sigmoid',
                         kernel_initializer=initializers.RandomNormal(stddev=0.1),
                         bias_initializer=initializers.RandomNormal(stddev=0.1))
        ])

        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.nn_model.load_weights(latest_checkpoint_path)

    def get_output(self, x):
        return self.nn_model(x)
    
    def train(self):
        validation_interval = 500
        episodes = 5000
        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                self.test(episodes=100)

            player_agents = [AIAgent('x', self), AIAgent('o', self)]
            game = Game()

            player_num = random.randint(0, 1)
            game.current_player_token = game.player_tokens[player_num]

            game_step = 0
            while not game.is_finished():
                observed_state = game.extract_features()
                state_value = player_agent.ai_model.get_output(observed_state)

                player_agent = player_agents[player_num]
                game.make_move(player_agent)

                player_num = (player_num + 1) % 2
                game.current_player_token = game.player_tokens[player_num]

                next_observed_state = game.extract_features()
                if (game.is_finished()):
                    if game.winner_token() == Game.TOKEN_X:
                        next_state_value = 1.0
                    elif game.winner_token() == Game.TOKEN_O:
                        next_state_value = 0.0
                    else:
                        next_state_value = 0.5
                else:
                    next_state_value = player_agent.ai_model.get_output(next_observed_state)

                # self.sess.run(self.train_op, feed_dict={ self.x: x, next_state_value })

                game_step += 1

            winner = game.winner()