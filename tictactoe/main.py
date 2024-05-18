from absl import app
from absl import flags
import os
import tensorflow as tf

from model import Model

from game.game import Game
from game.agents.human_agent import HumanAgent
from game.agents.random_agent import RandomAgent
from game.agents.ai_agent import AIAgent

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'play', 'List of modes: play, test, train.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from latest checkpoint.')

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

def main(argv):
    ai_model = Model(model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
    if FLAGS.mode == 'test':
        # TODO: add testing
        pass

    if FLAGS.mode == 'play':
        game = Game()
        player_agents = [HumanAgent('X'), AIAgent('O', ai_model)]
        game.play(player_agents, draw=True)
        pass

    if FLAGS.mode == 'train':
        # TODO: add training
        pass

if __name__ == '__main__':
    app.run(main)
