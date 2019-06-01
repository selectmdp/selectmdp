import random
import numpy as np
import tensorflow as tf
import copy as cp

# from q_network import config # running this file
# from q_network.q_network import Q_Network # running this file

FLAGS = tf.flags.FLAGS

class Agent:
    """
    Agent for choosing by greedy rule
    """
    def __init__(self, scope, session, FEATURE_SIZEs, main_params, target_params):
        self.sess = session
        self._init_parameters(scope, FEATURE_SIZEs, main_params, target_params)
        self._init_q_network()

    def _init_parameters(self, scope, FEATURE_SIZEs, main_params,target_params):
        """
        Initialize basic parameters from FLAGS or Inputs
        """
        # init parameters
        self.N_EQRT = FLAGS.N_EQUIVARIANT
        self.N_IVRT = FLAGS.N_INVARIANT
        self.N_CHOOSE = FLAGS.N_CHOOSE
        self.N_SUBACT = FLAGS.N_SUBACT
        self.scope = scope
        self.NWRK_EXPAND = FLAGS.NWRK_EXPAND

        # initialize parameers
        self.FEATURE_SIZEs = FEATURE_SIZEs
        self.main_params, self.target_params = main_params, target_params

        # print('init',self.main_params[0][0][0][-1][-1],  self.sess.run(self.main_params[0][0][0][-1][-1]))
        # exit(0)

        # initialize exploration
        self.RELOAD_EP = FLAGS.RELOAD_EP
        self.TRAIN_STEP = FLAGS.TRAIN_STEP
        self.TRAIN_EPISODE = FLAGS.TRAIN_EPISODE
        self.global_step = self.RELOAD_EP * self.TRAIN_STEP * self.TRAIN_EPISODE

        self.INIT_EPS = FLAGS.INIT_EPS
        self.FINAL_EPS = FLAGS.FINAL_EPS
        self.EPS_DIS_CONST = 1. - 8. / (FLAGS.TRAIN_STEP * FLAGS.TRAIN_EPISODE * FLAGS.TOTAL_RELOAD-FLAGS.BUFFER_SIZE)
        # considering reload
       
        self.eps = np.max([self.FINAL_EPS, self.INIT_EPS * np.power(self.EPS_DIS_CONST, self.global_step-FLAGS.BUFFER_SIZE)])

        print('step, eps', self.global_step, self.eps)


        # self.EPS_MINUS = 1./FLAGS.EPS_DIS_CONST

    def _init_q_network(self):
        """
        Initialize the q_network
        """
        if FLAGS.AGENT == "dqn_ind":
            if FLAGS.NETWORK == "DIFFEINET":
                from agents.q_network.q_network_ind import Q_Network         
            elif FLAGS.NETWORK == "SHAREEINET":
                from agents.q_network.q_network_ind import Q_Network
            elif FLAGS.NETWORK == "PROGRESSIVE":
                from agents.q_network.q_network_ind import Q_Network
            elif FLAGS.NETWORK == "PROGRESSIVE_1_K":
                from agents.q_network.q_network_ind import Q_Network   
                
            elif FLAGS.NETWORK == "PROGRESSIVE_ROOT":
                from agents.q_network.q_network_ind import Q_Network
            else:
                print('wrong network type')
                exit(0)

        elif FLAGS.AGENT == "dqn" or FLAGS.AGENT =="CENTRAL_GREEDY" or FLAGS.AGENT =="dqn_myopic" or FLAGS.AGENT =="dqn_individual":
            from agents.q_network.q_network_ind import Q_Network
            if FLAGS.NETWORK == "DIFFEINET":
                from agents.q_network.q_network import Q_Network
            elif FLAGS.NETWORK == "SHAREEINET":
                from agents.q_network.q_network import Q_Network
            elif FLAGS.NETWORK == "PROGRESSIVE":
                from agents.q_network.q_network import Q_Network
            elif FLAGS.NETWORK == "PROGRESSIVE_1_K":
                from agents.q_network.q_network import Q_Network
            elif FLAGS.NETWORK == "PROGRESSIVE_ROOT":
                from agents.q_network.q_network import Q_Network
            elif FLAGS.NETWORK == "VANILLA":
                from agents.q_network.q_network_vanilla import Q_Network

        else:
            print('wrong agent')
            exit(0)

        self.q_network \
            = Q_Network(self.main_params, self.target_params, self.FEATURE_SIZEs, self.sess)


    def _add_trans_to_qnetwork(self, obs_state, act, reward, obs_next_state):
        """
        Add trans to Q_network. Fill up the replay buffer
        obs_state = [in, eq-unsel]
        act = [array]
        reward = r
        obs_next_state = [in, eq-unsel]
        Return trans =[in, eq-unsel, act, reward, in, eq-unsel]
        """
        trans = []
        for state in obs_state:
            trans.append(state)
        
        trans.append(act)
        # print('act')
        trans.append(reward)

        for next_state in obs_next_state:
            trans.append(next_state)

        self.q_network.add_trans_to_buffer(trans)
    
    def _receive_state(self, obs):
        """
        Receive the observation
        """
        return

    def act(self, obs, train=True):
        """
        Choose actions
        """

        self.obs_state = []
        # if self.N_IVRT:
        self.obs_state.append(obs['invariant_array'])
        self.obs_state.append(obs['equivariant_array'])
        
        # random policy
        if train: #train mode
            if self.global_step < self.q_network.BUFF_SIZE:
                # self.actions are [[a1,a2..,ak], [sub1,sub2,..,subk]] 
                action_selects = random.sample(range(self.N_EQRT), self.N_CHOOSE)
                action_subsacts = np.random.randint(self.N_SUBACT, size=self.N_CHOOSE)
                action_subsacts = action_subsacts.tolist()

            # choice
            else:
                action_selects, action_subsacts = self.q_network.sample_acts(obs, self.eps)

        else: # Test mode
            action_selects, action_subsacts = self.q_network.sample_acts(obs, 0)

        # append actions
        self.actions = []
        self.actions.append(action_selects)
        self.actions.append(action_subsacts)

        # change the actions with proper form
        if FLAGS.ENVIRONMENT =='circle_env':
            actions = self.actions[0]  # only give first type of actions

        if FLAGS.ENVIRONMENT =='predator_prey': # prey and predator
            actions = -1 * np.ones(self.N_EQRT)
            # print('action_subac', action_subsacts) # for stay action
            actions[action_selects] = np.array(action_subsacts)+1 # without stay action
        if FLAGS.ENVIRONMENT=="predator_prey_discrete": # prey and predator with grid
            actions = []
            actions.append(action_selects)
            actions.append(action_subsacts)
            actions = np.array(actions)
        if FLAGS.ENVIRONMENT=="circle_env_good":
            actions = np.array(self.actions)
        return actions
    
    def receive_reward(self, reward):
        """
        get the reward
        """
        self.reward = reward

    def receive_next_state(self, obs, train=True):
        """
        Receive the next states = [in, eq]. Also add trans to buffer, increase the global, step. add, update the network...
        """
        self.obs_next_state = []
        # if self.N_IVRT:
        self.obs_next_state.append(obs['invariant_array'])
        self.obs_next_state.append(obs['equivariant_array'])

        if train: # update neural network for train mode
            
            self._process_after_receive()

    def _process_after_receive(self):
        """
        Do the process after receive the things.
        Update parameters and copy the parameters
        """
        # print('experience',self.obs_state, self.actions, self.reward, self.obs_next_state)
        self._add_trans_to_qnetwork(self.obs_state, self.actions, self.reward, self.obs_next_state
        )
        # 
        # print('self.global_step',self.global_step)
        if self.global_step >= self.q_network.BUFF_SIZE:
            # print('before update',self.q_network.main_params[-1][0][0][-1][-1],  self.sess.run(self.q_network.main_params[-1][0][0][-1][-1]))

            self.q_network.update_network()
            # print('after update',self.q_network.main_params[-1][0][0][-1][-1],  self.sess.run(self.q_network.main_params[-1][0][0][-1][-1]))
            # exit(0)

            self.eps = np.max([self.FINAL_EPS, self.eps * self.EPS_DIS_CONST])

            if self.global_step % FLAGS.UPDATE_TARGET == 0:
                self.q_network.copy_target()

        self._increase_step()
    
    def _increase_step(self):
        """
        Increase the step and decrease eps
        """
        self.global_step += 1

    def get_loss(self):
        losses = -0.015 * np.ones(self.N_CHOOSE)
        # if self.global_step > self.q_network.BUFF_SIZE+1:
        #     losses = self.q_network.get_losses()
        return losses

    def get_q_value(self):
        q_values = -0.015 * np.ones(self.N_CHOOSE)
        # if self.global_step > self.q_network.BUFF_SIZE+2:
        #     q_values = self.q_network.get_q_values()
        return q_values

    def copy_parameters(self):
        pass

    def save_network(self):
        pass
