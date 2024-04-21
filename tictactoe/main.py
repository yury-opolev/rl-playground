from absl import app
from absl import flags
import tensorflow as tf

from game.game import Game
from game.agents.human_agent import HumanAgent
from game.agents.random_agent import RandomAgent

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'play', 'List of modes: play, test, train.')

def main(argv):
    if FLAGS.mode == 'test':
        # TODO: add testing
        pass

    if FLAGS.mode == 'play':
        game = Game()
        player_agents = [HumanAgent('x'), RandomAgent('o')]
        game.play(player_agents, draw=True)
        pass

    if FLAGS.mode == 'train':
        # TODO: add training
        pass

if __name__ == '__main__':
    app.run(main)
