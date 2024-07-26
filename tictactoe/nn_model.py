import random
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

from pathlib import Path
from keras import layers
from keras import models
from keras import initializers

from game.env import Game
from game.agents.ai_agent import AIAgent
from game.agents.random_agent import RandomAgent
from replay_memory import ReplayMemory

class NNModel(object):
    def __init__(self):
        self.nn_model = models.Sequential([
            layers.Input(shape=(29,)),
            layers.Dense(29, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(17, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(9, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(1, activation=keras.activations.tanh,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05))
        ])

        self.learning_rate = 0.001
        self.gamma = 0.9
        self.lamda = 0.7
        self.batch_size = 64
        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.nn_model.compile(optimizer=self.optimizer, loss=keras.losses.MeanSquaredError())

    def get_action_features(self, action):
        x, y = action
        action_index = x * 3 + y
        action_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        action_features[action_index] = 1.0
        return action_features

    def get_output(self, state_features, action):
        action_features = self.get_action_features(action)
        input_state = tf.convert_to_tensor([np.concatenate((state_features, action_features), axis=None)])
        return self.__get_output_internal(input_state)
    
    @tf.function
    def __get_output_internal(self, input_state):
        return self.nn_model(input_state)

    def test(self, episodes=100):
        winners = { Game.EMPTYTOKEN: 0, Game.TOKEN_X: 0, Game.TOKEN_O: 0 }
        for episode in range(episodes):
            game = Game()

            if random.choice([0, 1]) == 0:
                player_agents = [AIAgent('X', self), RandomAgent('O')]
            else:
                player_agents = [RandomAgent('X'), AIAgent('O', self)]
            
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

    def train(self, episodes=10000, epsilon=0.5, validate=False):
        validation_interval = 10000
        for episode in range(episodes):
            if validate and (episode % validation_interval == 0):
                print(f"Testing after {episode} episodes:")
                self.test()
                print()

            player_agents = [AIAgent('X', self), AIAgent('O', self)]
            game = Game()

            current_player_agent = self.get_player_agent(game, player_agents)

            is_done = False
            while not is_done:
                # get state S 
                observed_state = game.extract_features()

                # get action A (and Q(S, A)) 
                actions = game.get_possible_actions()
                (action, action_value) = current_player_agent.get_action(actions, game, epsilon)

                # get R
                reward, is_done = game.step(action, game.grid, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

                # get S'
                next_observed_state = game.extract_features()

                # get action A' (and Q(S',A'))
                actions_next = game.get_possible_actions()
                (action_next, action_next_value) = current_player_agent.get_action(actions_next, game, epsilon)

                # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
                self.update_weights(observed_state, action, (reward + self.gamma * action_next_value) - action_value)

        print(f"Final testing:")
        self.test()
        print()

    def update_weights(self, state, expected_value, discount_rate=1.0):
        with tf.GradientTape() as tape:
            predicted_value = self.get_output(state)
            gradients = tape.gradient(predicted_value, self.nn_model.trainable_weights)

        for i, gradient in enumerate(gradients):
            weight = self.nn_model.trainable_weights[i]
            weight.assign_add(self.learning_rate * tf.reshape(expected_value - predicted_value, shape=(1,)) * gradient)

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
