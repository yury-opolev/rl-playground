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

        self.memory = ReplayMemory(10000)

    def get_actions_output(self, state_features):
        input_state = tf.convert_to_tensor([state_features])
        return self.nn_model(input_state)[0].numpy()

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

                self.memory.push([observed_state], action, reward, [next_observed_state], best_next_state_action_value)

                # Q(S, A) <- Q(S, A) + alpha * ((R  + gamma * Q(S',A')) - Q(S, A))
                self.optimize_model(batch_size=128)
                # self.update_weights(observed_state, action, reward, best_next_state_action_value)

        print(f"Final testing:")
        self.test()
        print()

    def optimize_model(self, batch_size: int):
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Вычислить маску нефинальных состояний и соединить элементы батча
        # (финальным состоянием должно быть то, после которого моделирование закончилось)

        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                   batch.next_state)), device=device, dtype=torch.bool)
        
        #non_final_next_states = torch.cat([s for s in batch.next_state
        #                                            if s is not None])
        
        # Собираем батчи для состояний, действий и наград
        state_batch = tf.concat(batch.state, axis=0)
        action_batch = tf.concat(batch.action, axis=0)
        reward_batch = tf.concat(batch.reward, axis=0)

        # Вычислить Q(s_t, a) - модель вычисляет Q(s_t), 
        # затем мы выбираем столбцы предпринятых действий. 
        # Это те действия, которые были бы предприняты для каждого состояния партии в соответствии с policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Вычислить V(s_{t+1}) для всех следующих состояний.
        # Ожидаемые значения действий для не_финальных_следующих_состояний вычисляются 
        # на основе "старшей" целевой_сети; выбирается их наилучшее вознаграждение с помощью max(1)[0].
        # Это объединяется по маске, так что мы будем иметь либо ожидаемое значение состояния, 
        # либо 0, если состояние было финальным.
        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Вычисляем ожидаемые Q значения
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Объединяем все в общий лосс
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Готовим градиент
        optimizer.zero_grad()
        loss.backward()
        # Обрезаем значения градиента - проблемма исчезающего/взрывающего градиента
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        
        optimizer.step()

    def update_weights(self, state, action, reward, next_state_action_value):
        with tf.GradientTape() as tape:
            predicted_value = self.get_output(state, action)
            gradients = tape.gradient(predicted_value, self.nn_model.trainable_weights)

        delta = (reward + self.gamma * next_state_action_value - predicted_value)
        for i, gradient in enumerate(gradients):
            weight = self.nn_model.trainable_weights[i]
            weight.assign_add(self.learning_rate * tf.reshape(delta, shape=(1,)) * gradient)

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
