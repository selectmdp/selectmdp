import tensorflow as tf
import random
import config
import numpy as np

FLAGS = tf.flags.FLAGS

class Agent:
    """
    Agent for choosing uniform random action 
    """
    # def __init__(self, scope, session = None, n_eqrt = FLAGS.N_EQUIVARIANT, n_ivrt = FLAGS.N_INVARIANT):
    def __init__(self, scope, session = None, feature_size=None, main_params=None, target_params=None):

        self.N_EQRT = FLAGS.N_EQUIVARIANT
        self.N_IVRT = FLAGS.N_INVARIANT
        self.N_CHOOSE = FLAGS.N_CHOOSE
        self.N_SUBACT = FLAGS.N_SUBACT

    def act(self, obs):
        """Choose dummy actions
        """
        if FLAGS.ENVIRONMENT == 'circle_env':
            action_list = random.sample(range(self.N_EQRT), self.N_CHOOSE)
        elif FLAGS.ENVIRONMENT == 'circle_env_good':
            action_selects = random.sample(range(self.N_EQRT), self.N_CHOOSE)
            action_subsacts = np.random.randint(self.N_SUBACT, size=self.N_CHOOSE)  
            actions = []          
            actions.append(action_selects)
            actions.append(action_subsacts)
            actions = np.array(actions)
            action_list = actions

        elif FLAGS.ENVIRONMENT == 'predator_prey':
            # -1: unselected, 0: stay, 1: left, 2: up, 3: right, 4: down
            action_selects = random.sample(range(self.N_EQRT), self.N_CHOOSE)
            action_subsacts = np.random.randint(self.N_SUBACT, size=self.N_CHOOSE)
            action_subsacts = action_subsacts.tolist()
            action_list = -1 * np.ones(self.N_EQRT)
            action_list[action_selects] = action_subsacts
        else:
            raise Exception('Undefined environment: {}'.format(FLAGS.ENVIRONMENT))

        return action_list
    
    def receive_next_state(self, obs):
        return None

    def receive_reward(self, reward):
        pass

    def increase_step(self):
        pass

    def get_loss(self):
        return np.zeros(self.N_CHOOSE)

    def get_q_value(self):
        return np.zeros(self.N_CHOOSE)

    def global_step(self):
        pass

    def copy_parameters(self):
        pass

    def save_network(self):
        pass
