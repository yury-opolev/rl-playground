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
from replay_memory import Transition, ReplayMemory

class NNModel(object):
    def __init__(self):
        self.nn_model = models.Sequential([
            layers.Input(shape=(18,)),
            layers.Dense(18, activation=keras.activations.leaky_relu,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05)),
            layers.Dense(9, activation=keras.activations.tanh,
                         kernel_initializer=initializers.RandomNormal(stddev=0.05),
                         bias_initializer=initializers.RandomNormal(stddev=0.05))
        ])

        self.learning_rate = 0.001
        self.gamma = 0.9
        self.lamda = 0.7
        self.batch_size = 64

        self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.nn_model.compile(optimizer=self.optimizer, loss=keras.losses.MeanSquaredError())

        self.target_nn_model = keras.models.clone_model(self.nn_model)
        self.target_nn_model.build(input_shape=(18,))
        self.target_nn_model.compile(optimizer=self.optimizer, loss=keras.losses.MeanSquaredError())
        self.target_nn_model.set_weights(self.nn_model.get_weights())

        self.memory = ReplayMemory(10000)

    def get_actions_output(self, state_features):
        input_state = tf.convert_to_tensor([state_features])
        return self.nn_model(input_state)[0].numpy()

    def get_action_index(self, action):
        x, y = action
        action_index = x * 3 + y
        return action_index

    def test(self, episodes=1000):
        ai_wins = { 'WON': 0, 'LOST': 0, 'DRAW': 0 }
        for episode in range(episodes):
            game = Game()

            if random.choice([0, 1]) == 0:
                player_agents = [AIAgent('X', self), RandomAgent('O')]
                ai_agent_token = Game.TOKEN_X
                random_agent_token = Game.TOKEN_O
            else:
                player_agents = [RandomAgent('X'), AIAgent('O', self)]
                ai_agent_token = Game.TOKEN_O
                random_agent_token = Game.TOKEN_X

            current_player_agent = self.get_player_agent(game, player_agents)

            while not game.is_finished():
                actions = game.get_possible_actions()
                action_value = current_player_agent.get_action(actions, game)
                action, value = action_value
                game.take_action(action, game.current_player_token)

                game.change_player()
                current_player_agent = self.get_player_agent(game, player_agents)

            winner_token = game.winner_token
            if winner_token is None or winner_token == Game.EMPTYTOKEN:
                ai_wins['DRAW'] = ai_wins['DRAW'] + 1
            else:
                if ai_agent_token == winner_token:
                    ai_wins['WON'] = ai_wins['WON'] + 1
                else:
                    ai_wins['LOST'] = ai_wins['LOST'] + 1

        print(f"AI draws: {ai_wins['DRAW']}, wins: {ai_wins['WON']}, losses: {ai_wins['LOST']}.")

    def train(self, episodes=10000, epsilon=0.5, validate=False):
        update_target_network_every = 250
        learn_every = 10
        min_buffer_size_to_learn = 1000

        global_steps = 0
        validation_interval = 1000
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
                global_steps += 1

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
                if is_done:
                    best_next_state_action_value = 0.0
                else:
                    actions_next = game.get_possible_actions()
                    (best_next_action, best_next_state_action_value) = current_player_agent.get_action(actions_next, game, 0.0)

                self.memory.push([observed_state], action, reward, [next_observed_state], best_next_state_action_value, (current_player_agent.player_token != Game.TOKEN_X))

                # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
                if global_steps % learn_every == 0:
                    self.optimize_model(batch_size=128)

                if global_steps % update_target_network_every == 0:
                    self.optimize_model(batch_size=128)

            # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
            if global_steps % learn_every == 0 and len(self.memory) >= min_buffer_size_to_learn:
                self.optimize_model(batch_size=128)

            if global_steps % update_target_network_every == 0:
                self.update_target_network()

        print(f"Final testing:")
        self.test()
        print()

    def update_target_network(self):
        self.target_nn_model.set_weights(self.nn_model.get_weights())

    def optimize_model(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = tf.concat(batch.state, axis=0)

        expected_state_action_values = []
        for transition in transitions:
            current_output = self.get_actions_output(transition.state[0])
            expected_output = current_output.copy()
            action_index = self.get_action_index(transition.action)

            input_next_state = tf.convert_to_tensor([transition.next_state[0]])
            next_state_action_values = self.target_nn_model(input_next_state)[0].numpy()
            # TODO: remove illegal moves
            if transition.is_minimizing_value:
                next_state_value = np.min(next_state_action_values)
            else:
                next_state_value = np.max(next_state_action_values)

            expected_output[action_index] = transition.reward + self.gamma * next_state_value
            expected_state_action_values.append([expected_output])

        expected_state_action_values_batch = tf.concat(expected_state_action_values, axis=0)
        self.nn_model.fit(state_batch, expected_state_action_values_batch, verbose=0)

    def update_weights(self, state, action, reward, next_state_action_value):
        with tf.GradientTape() as tape:
            predicted_value = self.get_output(state, action)
            gradients = tape.gradient(predicted_value, self.nn_model.trainable_weights)

        delta = (reward + self.gamma * next_state_action_value - predicted_value)
        for i, gradient in enumerate(gradients):
            weight = self.nn_model.trainable_weights[i]
            weight.assign_add(self.learning_rate * tf.reshape(delta, shape=(1,)) * gradient)

    def restore_weights(self, path, target_path):
        weights_filepath = Path(path)
        if weights_filepath.exists():
            print(f'Restoring weights: {path}')
            self.nn_model.load_weights(path)
            self.target_nn_model.load_weights(target_path)

    def save_weights(self, path, target_path):
        print(f'Saving weights: {path}')
        self.nn_model.save_weights(path)
        self.target_nn_model.save_weights(target_path)

    def get_player_agent(self, game, player_agents):
        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]
        return current_player_agent
