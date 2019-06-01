import random
import numpy as np
import copy as cp
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# when test this file directly
# from replay_buffer import Replay_Buffer
# from params_einet import Params
# import config

# when test dqn.py
# from q_network.replay_buffer import Replay_Buffer
# from q_network.params_einet import Params

# when test train.py
from agents.q_network.replay_buffer_ind import Replay_Buffer
from agents.q_network.params_einet import Params

"""
class Params

_generate_param  --> single 2d matrix (in_feature, out_feature)
_generate_layer  --> matrices needed for a single layer [Wg[y][z], Ws[y], b[y]]
generate_layers  --> matrices needed for x layers [Wg[x][y][z], Ws[x][y], b[x][y]]

                 ==> params = [Wg, Ws, b]
                     where each element of W is a matrix [in_feature, out_feature]

class Q_Network

_extend_params_to_network  --> placeholders, Q_network
_generate_Q_network        --> main and target
_generate_networks         --> N_CHOOSE

"""
class Q_Network(object):
    """
    All related to Q-Networks is here.
    Generate N_CHOOSE number of Q-Networks (whether shared params or DIFFEINET)
    """
    def __init__(self, main_params, target_params, FEATURE_SIZEs, sess):
        """
        :param main_params: [Wg, Ws, b] * k
        :param target_params: [Wg, Ws, b] * k
        :param FEATURE_SIZEs: [[3, 5, 7], [300, 5, 14], [30, 5000, 7]]
        :param sess: tf.Session()

        [Wg, Ws, b] = Params('global', FEATURE_SIZEs, ELEMENT_SIZES)
        Wg[x][y][z]
        Ws[x][y]
        b[x][y]
        """

        # Part 1
        # set class attributes
        self.sess = sess
        self._init_parameters(FEATURE_SIZEs)
        self._get_params_einet(main_params, target_params)

        # Part 2
        self._generate_networks()  # generate placeholders and data flows
        self._generate_placeholders()  # generate placeholders for updating everything
        self._generate_replay_buffers()

        # Part 3
        # generate the data flows for q-values losses (for printing)
        self._generate_q_values()
        self._generate_losses()
        self._generate_optimizer()

    # ------------------------------------------ Part 1 ------------------------------------------ #

    def _init_parameters(self, FEATURE_SIZEs):
        """
        Initialize basic parameters from FLAGS or Inputs
        """
        # receive parameters related to env
        self._generate_env_parameters()
        # receive parameters related to agent
        self._generate_rl_parameters(FEATURE_SIZEs)

    def _get_params_einet(self, main_params, target_params):
        """
        main_params: shared parameters for main_network (to updated with gradient descent)
        target_params: shared parameters for target_network (stop gradient)

        params = [a(1), a(2),..,a(N_CHOOSE)] each ai denotes the params for q-network for ith transitions
        """
        # check whether params has correct form
        self._check_params(main_params)
        self._check_params(target_params)

        # set params
        self.main_params = main_params
        self.target_params = target_params


    def _check_params(self, params):
        """
        Check given params before generate Q-Networks for each ith transition
        """
        pass

    def _generate_env_parameters(self):
        """
        Initialize the parameters related to environment
        N_EQRT, N_IVRT, N_CHOOSE, FEATURE DIMs...
        """
        # number of elements and actions
        self.N_EQRT = FLAGS.N_EQUIVARIANT
        self.N_IVRT = FLAGS.N_INVARIANT
        self.N_CHOOSE = FLAGS.N_CHOOSE
        self.N_SUBACT = FLAGS.N_SUBACT

    def _generate_rl_parameters(self, FEATURE_SIZEs):
        """
        Initialize the paramteres related to RL or DQNs
        """
        self.GAMMA = FLAGS.GAMMA

        # reinforcement learning parameters
        self.BUFF_SIZE = FLAGS.BUFFER_SIZE
        self.BATCH_SIZE = FLAGS.BATCH_SIZE
        self.LR = FLAGS.LEARNING_RATE
        self.UPDATE_TARGET = FLAGS.UPDATE_TARGET
        self.ADAM_EP = FLAGS.ADAM_EP

        # related to exploration rate
        # self.global_step = 0
        # self.eps = FLAGS.INIT_EPS
        # self.eps_dis = 1. - 1./(FLAGS.EPS_DIS_CONST)

        # strings for easy auto completion
        # network parameters array size einets (expand_rate already used in params)
        self.LEAKY_RELU = FLAGS.REL_ALP
        self.NETWORK_STRING = ("in", "equi_selected", "equi-unselected")
        self.NETWORK_TYPE = len(self.NETWORK_STRING)  # number of element
        [self.IN_NET, self.EQ_SEL_NET, self.EQ_UNSEL_NET] = range(self.NETWORK_TYPE)

        # feature of the elements
        self.FEATURE_SIZEs = FEATURE_SIZEs
        self.N_LAYERS = len(self.FEATURE_SIZEs)

        # string and int parameters for auto completion
        # the placeholders also made by this form
        self.BUF_STRING = (
        "in", "equi_selected", "equi_unselected", "act_select", "act_subact", "reward", "next_in", "next_eq_selected",
        "next_eq_unselected")
        self.BUF_TYPE = len(self.BUF_STRING)  # number of the buffer type
        [self.IN_BUF, self.EQ_SEL_BUF, self.EQ_UNSEL_BUF, self.ACT_SEL_BUF, self.ACT_SUB_BUF, self.REWARD_BUF,
         self.NEXT_IN_BUF, self.NEXT_EQ_SEL_BUF, self.NEXT_EQ_UNSEL_BUF] = range(self.BUF_TYPE)

    # ------------------------------------------ Part 2 ------------------------------------------ #

    def _generate_networks(self):
        """
        Generate data flows for N_CHOOSE q-networks for each ith transitions.
        Generate placeholders for replay buffers.
        """
        # generate main and target Q_network array for ith transition = 0 ~ N_CHOOSE-1. placeholder has the same type with replay buffer.
        #TODO: erase N_CHOOSE


        # make the network and placeholders for (inv, eq-sel, eq-unsel)
        #TODO only one time
        self._generate_Q_network()

    def _generate_Q_network(self):
        """
        Generate each q-network for each ith transitions. for proper size and also generate placeholders either
        self.main_placeholders = (in, eq-sel, eq-unsel) placeholders for main q-network
        self.target_placeholders = (in, eq-sel, eq-unsel) placeholders for target q-network
        """
        element_sizes = (self.N_IVRT, 0, self.N_EQRT)

        # print('init',self.main_params[0][0][0][-1][-1],  self.sess.run(self.main_params[0][0][0][-1][-1]))
        # exit(0)
        # TODO:only one time
        self.main_placeholders, self.main_Qs \
            = self._extend_params_to_network(self.main_params[0], element_sizes)

        self.target_placeholders, self.target_Qs \
            = self._extend_params_to_network(self.target_params[0], element_sizes)

        self.target_Qs = tf.stop_gradient(self.target_Qs)

    def _extend_params_to_network(self, params, element_sizes):
        """
        Extend the params as the network and placeholders
        params[0]= Wg, params[1] = Ws, params[2] = b
        layers[a][0]: ath layers invariant elements
        layers[a][1]: ath layers eq-sel elements
        layers[a][2]: ath layers eq-unsel elements
        """
        # TODO:carefully about eq+sel types
        # NOTE:element_sizes[1]=0
        element_types = len(element_sizes)
        # = 3
        # placeholders = [None, None, None]
        placeholders = [None] * element_types

        # placeholders = [Place, Place, Place]
        # Place shape = (?, e, f)
        for element in range(element_types):
            placeholders[element] \
                = tf.placeholder(tf.float32, [None, element_sizes[element], self.FEATURE_SIZEs[0][element]])

        layers = []

        # layers = [[None, None, None],
        #           [None, None, None],
        #           [None, None, None],
        #           [None, None, None]]
        for _ in range(self.N_LAYERS):
            layers.append([None] * element_types)

        # layers = [[Place, Place, Place],
        #           [None, None, None],
        #           [None, None, None],
        #           [None, None, None]]
        for element in range(element_types):
            layers[0][element] = placeholders[element]

        Wg, Ws, b = params

        # generate NN before last layer
        for layer in range(self.N_LAYERS - 1):
            for element in range(element_types):
                # element = 0 invariant / 1 eq-selectable / 2 eq-unselectable

                # specific
                #
                # Ws = [[inv2inv, eqs2eqs, equs2equs]
                #       [inv2inv, eqs2eqs, equs2equs]
                #       [inv2inv, eqs2eqs, equs2equs]]
                # each element is 2d matrix shape = (in_feature, out_feature)
                #
                # temp_s: (?, e, in_f) * (in_f, out_f) = (?, e, out_f)
                # tensordot is cool
                #
                temp_s = tf.tensordot(layers[layer][element], Ws[layer][element], axes=[[-1], [0]])

                # general
                temp_g = 0
                element_size = 0
                for element_g in range(element_types):
                    element_size += element_sizes[element_g]

                for element_g in range(element_types):
                    if element_sizes[element_g]:
                        ## previous version
                        temp_reduced = tf.reduce_mean(layers[layer][element_g], axis=1)
                        temp_g += tf.tensordot(temp_reduced, Wg[layer][element_g][element], axes=[[-1], [0]])
                        # new version
                        # temp_reduced = 1/float(element_size) * tf.reduce_sum(layers[layer][element_g], axis=1)
                        # temp_g += tf.tensordot(temp_reduced, Wg[layer][element_g][element], axes=[[-1], [0]])

                temp_g = tf.reshape(temp_g, [-1, 1, self.FEATURE_SIZEs[layer + 1][element]])

                # biased
                temp_b = b[layer][element]
                temp_b = tf.reshape(temp_b, [-1, 1, self.FEATURE_SIZEs[layer + 1][element]])

                # layers = [[Place, Place, Place],
                #           [s+b+g, s+b+g, s+b+g],
                #           [s+b+g, s+b+g, s+b+g],
                #           [s+b+g, s+b+g, s+b+g]]
                # (single element in each loop)
                layers[layer + 1][element] = temp_s + temp_g + temp_b

                if not layer == self.N_LAYERS-2:
                    layers[layer + 1][element] = tf.nn.leaky_relu(layers[layer + 1][element], alpha=self.LEAKY_RELU)

        # only consider eq-unsel
        Q_network = tf.squeeze(layers[-1][-1])

        # output layer shape = (equi_unselected elements, equi_unselected features)
        Q_network = tf.reshape(Q_network, [-1, element_sizes[-1], self.FEATURE_SIZEs[-1][-1]])

        return placeholders, Q_network

    def _generate_placeholders(self):
        """
        Generate placeholders for each ith transition: the holder types are the same as replaybuffer
        #TODO: equi-sel erase
        self.placeholders[ith] = ["in","equi_selected" ,  "equi_unselected", "action_elements", "act_subact", "reward", "next_in", "next_equi_selected" , "next_eq_unselected"] for ith transition
        """
        self.placeholders = [None] * self.BUF_TYPE
        # Main: current invariant:0, eq-sel:1, eq-unsel:2 from Main_placeholders
        for buftype in range(3):
            self.placeholders[buftype] = self.main_placeholders[buftype]

        # Target: next inv :self.BUF_TYPE-3, eq-sel: self.BUF_TYPE-2, eq-unsel: self.BUF_TYPE-1
        for buftype in range(self.BUF_TYPE - 3, self.BUF_TYPE):
            self.placeholders[buftype] = self.target_placeholders[buftype - self.BUF_TYPE + 3]

        # set actions placeholder
        self.placeholders[self.ACT_SEL_BUF] = [None] * self.N_CHOOSE
        self.placeholders[self.ACT_SUB_BUF] = [None] * self.N_CHOOSE

        for ith in range(self.N_CHOOSE):
            self.placeholders[self.ACT_SEL_BUF][ith] = tf.placeholder(tf.float32, shape=[None, self.N_EQRT])
            self.placeholders[self.ACT_SUB_BUF][ith] = tf.placeholder(tf.float32, [None, self.N_SUBACT])

        # set reward placeholder
        self.placeholders[self.REWARD_BUF] = tf.placeholder(tf.float32, [None, 1])

    def _generate_replay_buffers(self):
        """
        Generate N_CHOOSE q-networks for each ith transitions.
        Since there are N_CHOOSE types of transitions, the placeholders are length k array
        """
        #TODO _change name
        self.replay_buffer = Replay_Buffer(self.BUFF_SIZE, self.BATCH_SIZE, self.N_EQRT, self.N_IVRT, self.N_CHOOSE, self.N_SUBACT,
                                           self.FEATURE_SIZEs[0][0], self.FEATURE_SIZEs[0][-1])

    # ------------------------------------------ Part 3 ------------------------------------------ #
    def _find_top_k_qs(self):
        """
        Generate top k q-values of the q-matrix (BATCH * N * C) and with following act-sel, act-sub values 
        """
        pass
    
    def _one_hot_seperate(self):
        """Seperate an one-hot vector with multiple hots into multiple one hots 
        """

    def _generate_q_values(self):
        """
        Generate q-values[ith] (data flow) for ith transition with the proper obs = (in, eq-sel, eq-unsel) as well as argmax
        """
        # q-values to make the data flow of loss
        self.main_q_values = [None] * self.N_CHOOSE
        self.target_q_values = [None] * self.N_CHOOSE
        self.main_Qs_1d = [None] * self.N_CHOOSE
        self.main_argmax_q = [None] * self.N_CHOOSE  # action for each transitions

        # action to get the q-values
        for ith in range(self.N_CHOOSE):
            self.main_Qs_1d[ith] = tf.reshape(self.main_Qs[ith], [-1, ])
        self.main_max_q = [None] * self.N_CHOOSE  # tf data-structure
        self.main_q_values_print = np.zeros(self.N_CHOOSE)  # numpy-to print

        # action values
        for ith in range(self.N_CHOOSE):
            # q-values and target values to generate losses
            # considering selecting elements
            self.main_q_values[ith] = tf.reduce_sum(tf.multiply(self.main_Qs, tf.reshape(self.placeholders[self.ACT_SEL_BUF][ith], [-1, self.N_EQRT, 1])), axis=1)
            # print(ith,tf.shape(self.main_q_values[ith]))

            # considering subaction
            self.main_q_values[ith] = tf.reduce_sum(tf.multiply(
                self.main_q_values[ith], self.placeholders[self.ACT_SUB_BUF][ith]), axis=1)

            # generate target value
            ## TODO centralized round robin need to change
            
            # print(ith,tf.shape(self.main_q_values[ith]))
        
            # print(tf.shape(m))
            self.target_q_values[ith] = tf.reduce_sum(tf.multiply(self.target_Qs, tf.reshape(self.placeholders[self.ACT_SEL_BUF][ith], [-1, self.N_EQRT, 1])), axis=1)

            # print(ith,tf.shape(self.target_q_values[ith]))

            # considering argmax
            self.target_q_values[ith] = tf.reduce_max(
                self.target_q_values[ith], axis=1)
            self.target_q_values[ith] = tf.reshape(self.placeholders[self.REWARD_BUF],[-1, ]) + self.GAMMA * self.target_q_values[ith]
            # print(ith,tf.shape(self.target_q_values[ith]))


            ## NOTE dqn case
        return None

    def _generate_losses(self):
        """
        Generate data flow of N_CHOOSE losses for each ith transitions.
        and also make the total_loss = loss_1 + loss_2 + ... + loss_N_CHOOSE
        """
        # self.losses = [None] * self.N_CHOOSE
        self.total_loss = 0
        for ith in range(self.N_CHOOSE): #TODO : erase
            # TODO: differeentiate when qmix or not
            # seperate losses
            # self.losses[ith] = tf.reduce_mean(tf.square(self.main_q_values[ith]
            # - self.target_q_values[ith]))

            # generate total loss #TODO:erase ith
            self.total_loss += tf.reduce_mean(tf.square(self.main_q_values[ith]
                                                        - self.target_q_values[ith]))

    def _generate_optimizer(self):
        """
        Generate the optimizer of the networks and copy optimizers
        """
        self.optimizer = tf.train.AdamOptimizer(self.LR, epsilon=self.ADAM_EP)
        self.train_step = self.optimizer.minimize(self.total_loss)
        self.copy_main_to_target = self._get_copy_ops('global_main', 'global_target')


    # ------------------------------------------ Externally usable ------------------------------------------ #

    def add_trans_to_buffer(self, trans):
        """
        Get the trans and fill up the replay buffer
        """
        self.replay_buffer.add_trans(trans)

    def sample_acts(self, obs, eps):
        ## TODO: check independent
        """
        Select the elements with the array act =  [a_0, a_1, ..., a_NCHOOSE-1]. Choose the a_1 element for second selection...
        obs_ith = [invariant, eq-select * ith, eq * (N_EQ-ith)] where each are the numpy matrices
        """
        act_select = [None] * self.N_CHOOSE
        act_subact = [None] * self.N_CHOOSE
        # TODO simiar to here

        # print('',self.FEATURE_SIZEs[0][self.EQ_SEL_BUF])
        obs_cooked = [obs['invariant_array'], np.zeros([0, self.FEATURE_SIZEs[0][self.EQ_SEL_BUF]]),
                      obs['equivariant_array']]

        temp_obs = []
        for obs in obs_cooked:
            if not np.shape(obs)[0] * np.shape(obs)[1] == 0:
                temp_obs.append(np.reshape(obs, [-1, np.shape(obs)[0], np.shape(obs)[1]]))
            else:
                temp_obs.append(np.ones([1, np.shape(obs)[0], np.shape(obs)[1]]))

        # just take cafe for slicing
        self.temp_sliced_act_que = np.arange(self.N_EQRT)

        feed_dict_act = dict(zip(self.main_placeholders, temp_obs))
        Q_values = self.sess.run(self.main_Qs, feed_dict_act)
        Q_values = np.reshape(Q_values, [-1, self.N_SUBACT])

        for ith in range(self.N_CHOOSE):
            Q_values, act_sel_ith, act_sub_ith =self._sample_act_ith(Q_values, eps, ith)

            # re-order the selected act order (act_ith is just for sliced obs)
            act_select[ith] = act_sel_ith
            act_subact[ith] = act_sub_ith

        return act_select, act_subact
    
    def _sample_act_ith(self, Q_values, eps=0, ith=0):
        """
        Samples the ith transitions of the states. Return the next form of obs = [in, eq-sel, eq-unsel]. It use main network.
        """
        next_Q_values = cp.deepcopy(Q_values)

        if np.random.rand(1) > eps:

            act_preargmax = np.argmax(Q_values) 

            # two types of actions
            act_sel_ith = int(act_preargmax / self.N_SUBACT)
            act_sub_ith = act_preargmax % self.N_SUBACT
                
        else:  # random search
            act_sel_ith = np.random.randint(0, self.N_EQRT - ith)
            act_sub_ith = np.random.randint(self.N_SUBACT)

        # action for return
        next_act_sel = self.temp_sliced_act_que[act_sel_ith]
        next_act_sub = act_sub_ith

        # delete some elements iin the matrix and ques
        next_Q_values = np.delete(next_Q_values, act_sel_ith, axis=0)
        self.temp_sliced_act_que = np.delete(self.temp_sliced_act_que, act_sel_ith, axis=0)

        return next_Q_values, next_act_sel, next_act_sub

    def update_network(self):
        """Update neural network by minimizing the loss
        """
        # making dictionary for placeholders whole update
        self.feed_dict_buf = {}
        # self.feed_dict_list = []
        batches = self.replay_buffer.get_batch()
        for buf_type in range(self.BUF_TYPE):
            if buf_type == self.ACT_SEL_BUF or buf_type == self.ACT_SUB_BUF:
                for ith in range(self.N_CHOOSE):
                    # self.feed_dict_list.append(self.placeholders[buf_type][ith])
                    self.feed_dict_buf[self.placeholders[buf_type][ith]] = batches[buf_type][ith]
            else:
                # self.feed_dict_list.append(self.placeholders[buf_type])
                self.feed_dict_buf[self.placeholders[buf_type]] = batches[buf_type]
        self.sess.run(self.train_step, self.feed_dict_buf)

        # print(self.feed_dict_buf)
        # for buf_type, trans_type in zip(self.placeholders, self.replay_buffer.get_batch()):
        #     if hasattr(buf_type, "__iter__") and  hasattr(trans_type, "__iter__"):
        #         print(buf_type)
        #         print(trans_type)
        #         for ith, trans_ith in zip(buf_type, trans_type):
        #             print(trans_ith)
        #             self.feed_dict_buf[ith] = trans_ith
        #     else:
        #         self.feed_dict_buf[buf_type] = trans_type
        # c = tf.constant([1])
        # self.sess.run(c, self.feed_dict_buf)
        # print(self.sess.run(self.feed_dict_list, self.feed_dict_buf))
        # print(self.sess.run(self.main_q_values, self.feed_dict_buf))
        # for ith in range(self.N_CHOOSE):
        # self.sess.run(self.target_q_values, self.feed_dict_buf)


    def copy_target(self):
        """
        Copy parameters from target to main
        """
        self.sess.run([self.copy_main_to_target])

    def _get_copy_ops(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def get_losses(self):
        """
        Get the current loss values and return it.
        """
        # print('main_q_values', self.sess.run(self.main_q_values, self.feed_dict_buf))
        # print('target_q',  self.sess.run(self.target_q_values, self.feed_dict_buf))
        # print('self.target_Qs[(ith + 1) self.N_CHOOSE]', self.sess.run(self.target_Qs[1], self.feed_dict_buf))
        # print('self.target_Qs[(ith+1)%self.N_CHOOSE]', self.sess.run(self.target_Qs, self.feed_dict_buf))
        # print('tf.multiply(self.main_Qs[ith]',
        #      self.sess.run([tf.multiply(self.main_Qs[2],tf.reshape(self.placeholders[2][self.ACT_SEL_BUF],[-1,self.N_EQRT-2,1])),tf.reshape(self.placeholders[2][self.ACT_SEL_BUF],[-1,self.N_EQRT-2,1])],self.feed_dict_buf))

        # print('losses', self.sess.run(tf.square(self.main_q_values[3]
        #     - self.target_q_values[3]),self.feed_dict_buf))
        # losses = self.sess.run(self.losses, feed_dict=self.feed_dict_buf)
        # return losses

    def get_q_values(self):
        """
        Return the q_values
        """
        return self.main_q_values_print


if __name__ == '__main__':
    FEATURE_SIZEs = [[3, 5, 7], [300, 5, 14], [30, 5000, 7]]
    ELEMENT_SIZES = [10, 20, 30]

    # main_params = [Wg, Ws, b] * k
    param = Params('global', FEATURE_SIZEs, ELEMENT_SIZES)
    main_params = [param.generate_layers()] * FLAGS.N_CHOOSE
    target_params = [param.generate_layers(False)] * FLAGS.N_CHOOSE

    sess = tf.Session()

    Q_network = Q_Network(main_params, target_params, FEATURE_SIZEs, sess)
    # print(Q_network.main_Qs[0])
    # print(Q_network.main_placeholders[0])

    x = [np.ones([1, FLAGS.N_INVARIANT, FEATURE_SIZEs[0][0]]),
         np.ones([1, 0, FEATURE_SIZEs[0][1]]),
         np.ones([1, FLAGS.N_EQUIVARIANT, FEATURE_SIZEs[0][2]])]
    # print(x[0], x[1], x[2])

    feed_dict = dict(zip(Q_network.main_placeholders[0], x))
    # print(Q_network.main_placeholders[0][0])

    sess.run(tf.global_variables_initializer())
    print('mhm', Q_network.main_placeholders[0][0])

    # feed_dict = {Q_network.main_placeholders[0][0]: x[0], Q_network.main_placeholders[0][1]: x[1], Q_network.main_placeholders[0][2]: x[2]}
    print(sess.run(Q_network.main_Qs[0], feed_dict))
    print(feed_dict)
