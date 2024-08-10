from absl import app
from absl import flags
import os
import tensorflow as tf
import random

print(">>> Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from nn_model import NNModel
from qtab_model import QTabModel
from qtab_model_stateval import QTabModelStateVal
from game.env import Game
from game.agents.human_agent import HumanAgent
from game.agents.random_agent import RandomAgent
from game.agents.ai_agent import AIAgent
from game.agents.ai_agent_stateval import AIAgentStateVal

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'play', 'List of modes: play, test, train, train_q, test_q.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from latest checkpoint.')
flags.DEFINE_boolean('save', False, 'If true, save the trained model (works only for q model and train mode).')
flags.DEFINE_string('savefileprefix', 'current', 'DQN model save file prefix.')

model_path = os.environ.get('MODEL_PATH', 'models/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

def main(argv):
    if FLAGS.mode == 'play':
        ai_model = NNModel()
        if FLAGS.restore:
            ai_model.restore_weights(f'models/{FLAGS.savefileprefix}.weights.h5', f'models/{FLAGS.savefileprefix}.target.weights.h5')

        game = Game()

        if random.choice([0, 1]) == 0:
            player_agents = [AIAgent('X', ai_model), HumanAgent('O')]
        else:
            player_agents = [HumanAgent('X'), AIAgent('O', ai_model)]

        if game.current_player_token == Game.TOKEN_X:
            current_player_agent = player_agents[0]
        else:
            current_player_agent = player_agents[1]

        while not game.is_finished():
            game.clear_screen()
            game.draw()

            actions = game.get_possible_actions()
            action_value = current_player_agent.get_action(actions, game)
            action, value = action_value
            game.take_action(action, game.current_player_token)

            game.change_player()
            if game.current_player_token == Game.TOKEN_X:
                current_player_agent = player_agents[0]
            else:
                current_player_agent = player_agents[1]

        game.clear_screen()
        game.draw()

        if game.winner_token == Game.TOKEN_X:
            print("X wins!")
        if game.winner_token == Game.TOKEN_O:
            print("O wins!")
        else:
            print("DRAW.")

    if FLAGS.mode == 'train':
        ai_model = NNModel()
        if FLAGS.restore:
            ai_model.restore_weights(f'models/{FLAGS.savefileprefix}.weights.h5', f'models/{FLAGS.savefileprefix}.target.weights.h5')

        print(f"Initial testing:")
        ai_model.test()
        print()

        batch_count = 1000
        for batch in range(batch_count):
            print(f"training batch: {batch} of {batch_count}")
            ai_model.train(episodes=10000, epsilon=0.9, validate=True)
            if FLAGS.save:
                ai_model.save_weights(f'models/{FLAGS.savefileprefix}.weights.h5', f'models/{FLAGS.savefileprefix}.target.weights.h5')

    if FLAGS.mode == 'test':
        ai_model = NNModel()
        if FLAGS.restore:
            ai_model.restore_weights(f'models/{FLAGS.savefileprefix}.weights.h5', f'models/{FLAGS.savefileprefix}.target.weights.h5')

        ai_model.test()

    if FLAGS.mode == 'q_play':
        qtab_model = QTabModel()
        if FLAGS.restore:
            qtab_model.restore_weights(f'models/{FLAGS.savefileprefix}.weights.qtab')

        game = Game()
        if random.choice([0, 1]) == 0:
            player_agents = [AIAgent('X', qtab_model), HumanAgent('O')]
        else:
            player_agents = [HumanAgent('X'), AIAgent('O', qtab_model)]

        current_player_agent = player_agents[0]
        while not game.is_finished():
            game.clear_screen()
            game.draw()

            actions = game.get_possible_actions()
            action_value = current_player_agent.get_action(actions, game)
            action, value = action_value
            game.take_action(action, game.current_player_token)

            game.change_player()
            if game.current_player_token == Game.TOKEN_X:
                current_player_agent = player_agents[0]
            else:
                current_player_agent = player_agents[1]

        game.clear_screen()
        game.draw()

        if game.winner_token == Game.TOKEN_X:
            print("X wins!")
        if game.winner_token == Game.TOKEN_O:
            print("O wins!")
        else:
            print("DRAW.")

    if FLAGS.mode == 'q_train':
        qtab_model = QTabModel()
        if FLAGS.restore:
            qtab_model.restore_weights('models/current.weights.qtab')

        print(f"Initial testing:")
        qtab_model.test()
        print()

        batch_count = 10000
        for batch in range(batch_count):
            print(f"training batch: {batch} of {batch_count}")
            (test_draw, test_win_x, test_win_o) = qtab_model.train(episodes=10000, epsilon=1.0)
            if FLAGS.save:
                qtab_model.save_weights('models/current.weights.qtab')

            if test_win_o == 0:
                print(">>> Extensive testing, as test results show 0 'O' win. <<<")
                (test_draw, test_win_x, test_win_o) = qtab_model.test(episodes=10000)
                if test_win_o == 0:
                    print(">>> Extensive test results show 0 'O' win, exiting training. <<<")
                    break

    if FLAGS.mode == 'q_test':
        qtab_model = QTabModelStateVal()
        if FLAGS.restore:
            qtab_model.restore_weights('models/current.weights.qtab')

        qtab_model.test(episodes=100000, print_failed_games=True)

    if FLAGS.mode == 'sval_play':
        svaltab_model = QTabModelStateVal()
        if FLAGS.restore:
            svaltab_model.restore_weights(f'models/sval-current.weights.qtab')

        game = Game()
        if random.choice([0, 1]) == 0:
            player_agents = [AIAgentStateVal('X', svaltab_model), HumanAgent('O')]
        else:
            player_agents = [HumanAgent('X'), AIAgentStateVal('O', svaltab_model)]

        current_player_agent = player_agents[0]
        while not game.is_finished():
            game.clear_screen()
            game.draw()

            actions = game.get_possible_actions()
            action_value = current_player_agent.get_action(actions, game)
            action, value = action_value
            game.take_action(action, game.current_player_token)

            game.change_player()
            if game.current_player_token == Game.TOKEN_X:
                current_player_agent = player_agents[0]
            else:
                current_player_agent = player_agents[1]

        game.clear_screen()
        game.draw()

        if game.winner_token == Game.TOKEN_X:
            print("X wins!")
        if game.winner_token == Game.TOKEN_O:
            print("O wins!")
        else:
            print("DRAW.")

    if FLAGS.mode == 'sval_train':
        svaltab_model = QTabModelStateVal()
        if FLAGS.restore:
            svaltab_model.restore_weights('models/sval-current.weights.qtab')

        print(f"Initial testing:")
        svaltab_model.test()
        print()

        batch_count = 10000
        for batch in range(batch_count):
            print(f"training batch: {batch} of {batch_count}")
            (test_draw, test_win_x, test_win_o) = svaltab_model.train(episodes=10000, epsilon=1.0)
            if FLAGS.save:
                svaltab_model.save_weights('models/sval-current.weights.qtab')

            if test_win_o == 0:
                print(">>> Extensive testing, as test results show 0 'O' win. <<<")
                (test_draw, test_win_x, test_win_o) = svaltab_model.test(episodes=10000)
                if test_win_o == 0:
                    print(">>> Extensive test results show 0 'O' win, exiting training. <<<")
                    break

    if FLAGS.mode == 'sval_test':
        svaltab_model = QTabModelStateVal()
        if FLAGS.restore:
            svaltab_model.restore_weights('models/sval-current.weights.qtab')

        svaltab_model.test(episodes=10000, print_failed_games=False)
 
if __name__ == '__main__':
    app.run(main)
