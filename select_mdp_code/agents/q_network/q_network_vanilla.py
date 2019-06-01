import random
import numpy as np
import copy as cp
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# when test this file directly
# from replay_buffer import Replay_Buffer
# from params_vanilla import Params
# import config

# when test dqn.py
# from q_network.replay_buffer import Replay_Buffer
# from q_network.params_einet import Params

# when test train.py
from agents.q_network.replay_buffer import Replay_Buffer
from agents.q_network.params_vanilla import Params

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

        # TODO: bathsize, learning rate, bufsize
        # reinforcement learning parameters
        self.BUFF_SIZE = FLAGS.BUFFER_SIZE
        self.BATCH_SIZE = FLAGS.BATCH_SIZE
        self.LR = FLAGS.LEARNING_RATE
        self.UPDATE_TARGET = FLAGS.UPDATE_TARGET
        self.ADAM_EP = FLAGS.ADAM_EP

        # related to exploration rate #TODO: it may be needed for test mode
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
        self.FEATURE_SIZEs = FEATURE_SIZEs  # [[3, 4, 3], [18, 24, 18], [18, 24, 18], [0, 0, 1]]
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
        self.main_Qs = [None] * self.N_CHOOSE
        self.target_Qs = [None] * self.N_CHOOSE
        self.main_placeholders = [None] * self.N_CHOOSE
        self.target_placeholders = [None] * self.N_CHOOSE

        # make the network and placeholders for (inv, eq-sel, eq-unsel)
        for ith in range(self.N_CHOOSE):
            self._generate_Q_network(ith)

    def _generate_Q_network(self, ith=0):
        """
        Generate each q-network for each ith transitions. for proper size and also generate placeholders either
        self.main_placeholders = (in, eq-sel, eq-unsel) placeholders for main q-network
        self.target_placeholders = (in, eq-sel, eq-unsel) placeholders for target q-network
        """
        element_sizes = (self.N_IVRT, ith, self.N_EQRT - ith)

        self.main_placeholders[ith], self.main_Qs[ith] \
            = self._extend_params_to_network(self.main_params, element_sizes, ith)
        # self.main_placeholders[ith], self.main_Qs[ith] \
        #     = self._extend_params_to_network(self.main_params[ith], element_sizes, self.N_LAYERS)

        self.target_placeholders[ith], self.target_Qs[ith] \
            = self._extend_params_to_network(self.target_params, element_sizes, ith)
        # self.target_placeholders[ith], self.target_Qs[ith] \
        #     = self._extend_params_to_network(self.target_params, element_sizes, self.N_LAYERS)

        self.target_Qs[ith] = tf.stop_gradient(self.target_Qs[ith])

    def _extend_params_to_network(self, params, element_sizes, transition):
        """
        Extend the params as the network and placeholders
        params[0]= Wg, params[1] = Ws, params[2] = b
        layers[a][0]: ath layers invariant elements
        layers[a][1]: ath layers eq-sel elements
        layers[a][2]: ath layers eq-unsel elements
        """

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

        # layers = [[None],
        #           [None],
        #           [None],
        #           [None]]
        for _ in range(self.N_LAYERS):
            layers.append([None])

        # layers = [[Flattened Place],
        #           [None],
        #           [None],
        #           [None]]
        # Flattend Place shape = (?, e0*f0 + e1*f1 + e2*f2)
        flattened_placeholder = [None] * element_types
        for element in range(element_types):
            # ph = placeholders[element]
            # shape = ph.get_shape().as_list()  # [None, e, f]
            # dim = np.prod(shape[1:])  # e*f
            # print("{}, {}".format(shape, dim))
            # flattened_placeholder[element] = tf.reshape(placeholders[element], [-1, dim])
            flattened_placeholder[element] = tf.layers.Flatten()(placeholders[element])
        flattened_placeholder = tf.concat(flattened_placeholder, axis=-1)
        # print(flattened_placeholder.get_shape().as_list())

        layers[0] = flattened_placeholder

        W, b = params[transition]

        # generate NN before last layer
        for layer in range(self.N_LAYERS - 1):

            # W = [layer2layer,
            #      layer2layer,
            #      layer2layer]
            # each element is 2d matrix shape = (in_dim, out_dim)
            #
            # temp_w: (?, in_dim) * (in_dim, out_dim) = (?, out_dim)
            temp_w = tf.tensordot(layers[layer], W[layer], axes=[[-1], [0]])

            # b = [layer2layer,
            #      layer2layer,
            #      layer2layer]
            # each element is 2d matrix shape = (1, out_dim)
            #
            # temp_b: (?, out_dim) -> (?, out_dim)
            temp_b = b[layer]
            temp_b = tf.reshape(temp_b, [-1, tf.shape(temp_b)[0], tf.shape(temp_b)[1]])

            layers[layer + 1] = temp_w + temp_b

            if not layer == self.N_LAYERS-2:
                layers[layer + 1] = tf.nn.leaky_relu(temp_w + temp_b, alpha=self.LEAKY_RELU)

        # output layer
        Q_network = tf.squeeze(layers[-1])

        # output layer shape = (equi_unselected elements, equi_unselected features)
        Q_network = tf.reshape(Q_network, [-1, element_sizes[-1], self.FEATURE_SIZEs[-1][-1]])

        return placeholders, Q_network

    def _generate_placeholders(self):
        """
        Generate placeholders for each ith transition: the holder types are the same as replaybuffer
        self.placeholders[ith] = ["in", "equi_selected", "equi_unselected", "action_elements", "act_subact", "reward", "next_in", "next_eq_selected", "next_eq_unselected"] for ith transition
        """
        self.placeholders = [None] * self.N_CHOOSE
        for ith in range(self.N_CHOOSE):
            # same as the types in replay buffer
            self.placeholders[ith] = [None] * self.BUF_TYPE

            # Main: current invariant:0, eq-sel:1, eq-unsel:2 from Main_placeholders
            for buftype in range(3):
                self.placeholders[ith][buftype] = self.main_placeholders[ith][buftype]

            # Target: next inv :self.BUF_TYPE-3, eq-sel: self.BUF_TYPE-2, eq-unsel: self.BUF_TYPE-1
            for buftype in range(self.BUF_TYPE - 3, self.BUF_TYPE):
                if not ith + 1 == self.N_CHOOSE:
                    self.placeholders[ith][buftype] = self.target_placeholders[ith + 1][buftype - self.BUF_TYPE + 3]
                else:
                    self.placeholders[ith][buftype] = self.target_placeholders[0][buftype - self.BUF_TYPE + 3]

            # set actions placeholder
            self.placeholders[ith][self.ACT_SEL_BUF] = tf.placeholder(tf.float32, shape=[None, self.N_EQRT - ith])
            self.placeholders[ith][self.ACT_SUB_BUF] = tf.placeholder(tf.float32, [None, self.N_SUBACT])

            # set reward placeholder
            self.placeholders[ith][self.REWARD_BUF] = tf.placeholder(tf.float32, [None, 1])

    def _generate_replay_buffers(self):
        """
        Generate N_CHOOSE q-networks for each ith transitions.
        Since there are N_CHOOSE types of transitions, the placeholders are length k array
        """
        self.replay_buffer = Replay_Buffer(self.BUFF_SIZE, self.BATCH_SIZE, self.N_EQRT, self.N_IVRT, self.N_CHOOSE, self.N_SUBACT,
                                           self.FEATURE_SIZEs[0][0], self.FEATURE_SIZEs[0][-1])


    # ------------------------------------------ Part 3 ------------------------------------------ #

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
        for ith in range(self.N_CHOOSE):
            # q-values and target values to generate losses
            # considering selecting elements
            self.main_q_values[ith] = tf.reduce_sum(tf.multiply(self.main_Qs[ith],                                                                tf.reshape(self.placeholders[ith][self.ACT_SEL_BUF],
                                                                           [-1, self.N_EQRT - ith, 1])), axis=1)
            # considering subaction
            self.main_q_values[ith] = tf.reduce_sum(tf.multiply(
                self.main_q_values[ith], self.placeholders[ith][self.ACT_SUB_BUF]), axis=1)

            # generate target value
            self.target_q_values[ith] = tf.reshape(self.placeholders[ith][self.REWARD_BUF],
            [-1, ]) + 1 * tf.reduce_max(self.target_Qs[(ith + 1) % self.N_CHOOSE], axis=[-2, -1])

            if ith + 1 == self.N_CHOOSE:  # for last transitions
                self.target_q_values[ith] = tf.reshape(self.placeholders[ith][self.REWARD_BUF],
                                                       [-1, ]) + self.GAMMA * tf.reduce_max(
                    self.target_Qs[(ith + 1) % self.N_CHOOSE], axis=[-2, -1])
                if FLAGS.AGENT=="dqn_myopic":
                    self.target_q_values[ith] = tf.reshape(self.placeholders[ith][self.REWARD_BUF],
                                                        [-1, ])

            # generate argmax_q values for fast computation
            self.main_max_q[ith] = tf.reduce_max(self.main_Qs[ith])
            # change with 1-d array to get pos of actions
            self.main_argmax_q[ith] = tf.argmax(self.main_Qs_1d[ith])

    def _generate_losses(self):
        """
        Generate data flow of N_CHOOSE losses for each ith transitions.
        and also make the total_loss = loss_1 + loss_2 + ... + loss_N_CHOOSE
        """
        # self.losses = [None] * self.N_CHOOSE
        self.total_loss = 0
        for ith in range(self.N_CHOOSE):
            # seperate losses
            # self.losses[ith] = tf.reduce_mean(tf.square(self.main_q_values[ith]
            # - self.target_q_values[ith]))

            # generate total loss
            self.total_loss += tf.reduce_mean(tf.square(self.main_q_values[ith]
                                                        - self.target_q_values[ith]))

    def _generate_optimizer(self):
        """
        Generate the optimizer of the networks
        """
        self.optimizer = tf.train.AdamOptimizer(self.LR, epsilon=self.ADAM_EP)
        self.train_step = self.optimizer.minimize(self.total_loss)

    # ------------------------------------------ Externally usable ------------------------------------------ #

    def add_trans_to_buffer(self, trans):
        """
        Get the trans and fill up the replay buffer
        """
        self.replay_buffer.add_trans(trans)

    def sample_acts(self, obs, eps):
        """
        Select the elements with the array act =  [a_0, a_1, ..., a_NCHOOSE-1]. Choose the a_1 element for second selection...
        obs_ith = [invariant, eq-select * ith, eq * (N_EQ-ith)] where each are the numpy matrices
        """
        # TODO: for multiple action it should be changed
        act_select = [None] * self.N_CHOOSE
        act_subact = [None] * self.N_CHOOSE

        # print('',self.FEATURE_SIZEs[0][self.EQ_SEL_BUF])
        obs_cooked = [obs['invariant_array'], np.zeros([0, self.FEATURE_SIZEs[0][self.EQ_SEL_BUF]]),
                      obs['equivariant_array']]

        # just take cafe for slicing
        self.temp_sliced_act_que = np.arange(self.N_EQRT)

        for ith in range(self.N_CHOOSE):
            obs_cooked, act_sel_ith, act_sub_ith = self._sample_act_ith(obs_cooked, eps, ith)

            # re-order the selected act order (act_ith is just for sliced obs)
            act_select[ith] = act_sel_ith
            act_subact[ith] = act_sub_ith

        return act_select, act_subact

    def _sample_act_ith(self, obs_ith, eps=0, ith=0):
        """
        Samples the ith transitions of the states. Return the next form of obs = [in, eq-sel, eq-unsel]. It use main network.
        """
        next_obs = cp.deepcopy(obs_ith)

        if np.random.rand(1) > eps:
            temp_obs = []
            for obs in obs_ith:
                if not np.shape(obs)[0] * np.shape(obs)[1] == 0:
                    temp_obs.append(np.reshape(obs, [-1, np.shape(obs)[0], np.shape(obs)[1]]))
                else:
                    temp_obs.append(np.ones([1, np.shape(obs)[0], np.shape(obs)[1]]))

            # dictionary for Q_networks
            feed_dict_act = dict(zip(self.main_placeholders[ith], temp_obs))
            self.main_q_values_print[ith], act_preargmax = self.sess.run(
                [self.main_max_q[ith], self.main_argmax_q[ith]], feed_dict_act)

            # two types of actions
            act_sel_ith = int(act_preargmax / self.N_SUBACT)
            act_sub_ith = act_preargmax % self.N_SUBACT

        else:  # random search
            act_sel_ith = np.random.randint(0, self.N_EQRT - ith)
            act_sub_ith = np.random.randint(self.N_SUBACT)

        # adding one-hot vector
        extend_sel = np.append(obs_ith[self.EQ_UNSEL_BUF][act_sel_ith],
                               self.replay_buffer._one_hot(self.N_SUBACT, act_sub_ith)).reshape([1, -1])

        # change the form of obs= (in, eq-sel, eq-unsel) for ith trans to (i+1)th trans
        next_obs[self.EQ_SEL_BUF] = np.append(obs_ith[self.EQ_SEL_BUF], extend_sel, axis=0)

        # action for return
        next_act_sel = self.temp_sliced_act_que[act_sel_ith]
        next_act_sub = act_sub_ith

        # delete some elements iin the matrix and ques
        next_obs[self.EQ_UNSEL_BUF] = np.delete(obs_ith[self.EQ_UNSEL_BUF], act_sel_ith, axis=0)
        self.temp_sliced_act_que = np.delete(self.temp_sliced_act_que, act_sel_ith, axis=0)

        return next_obs, next_act_sel, next_act_sub

    def update_network(self):
        """Update neural network by minimizing the loss
        """
        # making dictionary for placeholders whole update
        self.feed_dict_buf = {}
        for placeholder_ith, trans_ith in zip(self.placeholders, self.replay_buffer.get_batch()):
            for buf_type, trans_type in zip(placeholder_ith, trans_ith):
                self.feed_dict_buf[buf_type] = trans_type

        self.sess.run(self.train_step, self.feed_dict_buf)

    def copy_target(self):
        """
        Copy parameters from target to main
        """
        self.sess.run([self._get_copy_ops('global_main', 'global_target')])

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


    param = Params('global', FEATURE_SIZEs, ELEMENT_SIZES)

    main_params = [param.generate_layers()]
    target_params = [param.generate_layers(False)]

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
