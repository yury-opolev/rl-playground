import random
import tensorflow as tf
import keras
import copy
from pathlib import Path
from keras import layers
from keras import models
from keras import initializers

from game.env import Game
from game.agents.ai_agent import AIAgent
from game.agents.random_agent import RandomAgent

class NNModel(object):
    def __init__(self):
        self.nn_model = models.Sequential([
            layers.Input(shape=(20,)),
            layers.Dense(20, activation=keras.activations.sigmoid,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            # result is a prediction of probability of winning: 1 - 'X' wins, 0 - 'O' wins, 0.5 - draw
            layers.Dense(1, activation=keras.activations.tanh,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05))
        ])

        self.learning_rate = 0.01
        self.lamda = 0.7
        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)

    def init_eligiblity_trace(self):
        self.eligibility_traces = [tf.Variable(tf.zeros(weights.shape), trainable=False) for weights in self.nn_model.trainable_weights]

    def get_output(self, state):
        input_state = tf.convert_to_tensor([state])
        return self.nn_model(input_state)

    def test(self, episodes=100):
        winners = { Game.EMPTYTOKEN: 0, Game.TOKEN_X: 0, Game.TOKEN_O: 0 }
        for episode in range(episodes):
            game = Game()
            player_agents = [AIAgent('X', self), RandomAgent('O')]
            
            game.current_player_token = game.starting_random_player()
            current_player_agent = self.get_player_agent(game, player_agents)

            while not game.is_finished():
                actions = game.get_possible_actions()
                action_value = current_player_agent.get_action(actions, game)
                action, value = action_value
                game.take_action(action, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

            winner_token = game.winner_token
            if winner_token is None:
                winner_token = Game.EMPTYTOKEN
            winners[winner_token] = winners[winner_token] + 1

        print(f"Games played: {episodes}, draws: {winners[Game.EMPTYTOKEN]}, 'X' wins: {winners[Game.TOKEN_X]}, 'O' wins: {winners[Game.TOKEN_O]}.")

    def train(self, episodes=10000, epsilon=0.5):
        validation_interval = 100
        for episode in range(episodes):
            if episode % validation_interval == 0:
                print(f"Testing after {episode} episodes:")
                self.test(episodes=100)
                print()

            player_agents = [AIAgent('X', self), AIAgent('O', self)]
            game = Game()

            game.current_player_token = game.starting_random_player()
            current_player_agent = self.get_player_agent(game, player_agents)

            self.init_eligiblity_trace()

            while True:
                observed_state = game.extract_features()
                state_value = self.get_output(observed_state)

                actions = game.get_possible_actions()
                (action, action_value) = current_player_agent.get_action(actions, game, epsilon)

                reward, is_done = game.step(action, game.grid, game.current_player_token)

                next_observed_state = game.extract_features()
                next_state_value = self.get_output(next_observed_state)
                self.update_weights(observed_state, next_state_value)

                if is_done:
                    break

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

            self.update_weights(next_observed_state, reward)

        print(f"Final testing:")
        self.test(episodes=100)
        print()

    def update_weights(self, state, expected_value, discount_rate=1.0):
        with tf.GradientTape() as tape:
            predicted_value = self.get_output(state)
            gradients = tape.gradient(predicted_value, self.nn_model.trainable_weights)

        for i, gradient in enumerate(gradients):
            self.eligibility_traces[i].assign(self.lamda * self.eligibility_traces[i] + gradient)
            weight = self.nn_model.trainable_weights[i]
            weight.assign_add(self.learning_rate * tf.reshape(expected_value - predicted_value, shape=(1,)) * self.eligibility_traces[i])

    def restore_weights(self, path):
        weights_filepath = Path(path)
        if weights_filepath.exists():
            print(f'Restoring weights: {path}')
            self.nn_model.load_weights(path)

    def save_weights(self, path):
        print(f'Saving weights: {path}')
        self.nn_model.save_weights(path)

    def get_player_agent(self, game, player_agents):
        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]
        return current_player_agent
