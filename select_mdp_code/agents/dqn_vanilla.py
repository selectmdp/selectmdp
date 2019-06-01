import tensorflow as tf
import numpy as np
import os
import copy as cp
import config

FLAGS = config.flags.FLAGS

class Agent:

    def __init__(self, scope, session):
        self.N_EQRT = FLAGS.N_EQUIVARIANT
        self.N_IVRT = FLAGS.N_INVARIANT
        self.N_ELEMENTS = self.N_EQRT + self.N_IVRT
        self.N_FEATURES = FLAGS.N_FEATURES
        self.N_CHOOSE = FLAGS.N_CHOOSE
        self.OBS_DIM = self.N_ELEMENTS * self.N_FEATURES
        self.OUT_DIM = self.N_EQRT * self.N_CHOOSE

        self.scope = scope          # To check if this agent is 'global' or not
        self.test_mode = False      # Turn off test mode
        self.sess = session
        self.reset()

    def generate_networks(self):
        """
        Generate the neural networks
        """
        # Initialize replay buffer
        self.init_buffer()

        # Training parameters
        self.EPS_DIS_CONST = 1. - 8. / (FLAGS.TRAIN_STEP * FLAGS.TRAIN_EPISODE * FLAGS.TOTAL_RELOAD-FLAGS.BUFFER_SIZE)
        self.FINAL_EPS = FLAGS.FINAL_EPS
        self.eps = FLAGS.INIT_EPS
        self.lr = FLAGS.LEARNING_RATE

        # Parameters for main network Q1
        self.main_params = self.init_params(self.scope+'_main', trainable=True)
        # Main network Q1
        self.obs, self.Q1 = self.q_network(self.main_params)

        # Parameters for target network Q2
        self.targ_params = self.init_params(self.scope+'_target', trainable=False)
        # Target network Q2
        self.next_obs, self.Q2 = self.q_network(self.targ_params)
        self.Q2 = tf.stop_gradient(self.Q2)

        # Placeholder for action and reward
        self.action = tf.placeholder(tf.float32, [None, self.OUT_DIM])
        self.reward = tf.placeholder(tf.float32, [None, ])

        return None

    def generate_data_flow(self):
        """
        Generate data flow graph such as loss, train, etc.
        """
        # Output of main and target networks
        self.main_q_value = tf.reduce_sum(tf.multiply(self.Q1, self.action), axis=1)
        self.target_q_value = self.reward + FLAGS.GAMMA * tf.reduce_max(self.Q2, axis=1)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.main_q_value - self.target_q_value))

        # Get gradients from main network using loss
        # local_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope+'_main')
        # self.gradients = tf.gradients(self.loss, local_vars)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=FLAGS.ADAM_EP)

        # Operator to apply local gradients to global parameters
        # global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global')
        # self.apply_grads = self.optimizer.apply_gradients(zip(self.gradients, global_vars))

        # Training operator
        self.train_step = self.optimizer.minimize(self.loss)

        # Copy operators
        self.copy_ops = self.get_copy_ops(self.scope+'_main', self.scope+'_target')
        # self.copy_to_main_ops = self.get_copy_ops('global', self.scope+'_main')
        # self.copy_to_target_ops = self.get_copy_ops('global', self.scope+'_target')

    def reset(self):
        """
        Initialize network stuffs
        """
        self.global_step = 0

        # Layer sizes
        self.LAYER_SIZE = [self.OBS_DIM, self.OBS_DIM * FLAGS.NWRK_EXPAND, self.OBS_DIM * FLAGS.NWRK_EXPAND, self.OUT_DIM]

        # Initializations
        self.generate_networks()
        self.generate_data_flow()
        # if self.scope != 'global':
        #     self.generate_networks()
        #     self.generate_data_flow()
        # else:
        #     self.global_params = self.init_params('global', trainable=True)
        #     self.obs, self.Q_value = self.q_network(self.global_params)
        #     self.saver = tf.train.Saver(
        #         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global'))

    def get_copy_ops(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def save_network(self, path_name):
        """
        Save global network
        """
        self.saver.save(self.sess, path_name)

    def load_params(self, path_name):
        """
        Load trained parameters
        """
        self.saver.restore(self.sess, path_name)

    def init_params(self, name, trainable=True):
        """
        Initialize trainable parameters for Equi-invariant neural network
        """
        with tf.variable_scope(name):
            W, b = [], []
            for i in range(self.LAYER_SIZE.__len__() - 1):
                in_size = self.LAYER_SIZE[i]
                out_size = self.LAYER_SIZE[i+1]

                weight = tf.get_variable(name + '_W_' + str(i), [in_size, out_size], dtype=tf.float32,\
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
                W.append(weight)

                bias = tf.get_variable(name + '_b_' + str(i), [1, out_size], dtype=tf.float32,\
                    initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
                b.append(bias)

        return [W, b]

    def init_buffer(self):
        """
        Create replay buffer for observations, action, reward and next observations
        """
        self.exp_pointer = 0  # Buffer pointer
        self.eqrt_obs_que = np.empty([FLAGS.BUFFER_SIZE, self.N_EQRT, self.N_FEATURES])
        self.ivrt_obs_que = np.empty([FLAGS.BUFFER_SIZE, self.N_IVRT, self.N_FEATURES])
        self.act_que = np.empty([FLAGS.BUFFER_SIZE, self.OUT_DIM])
        self.rew_que = np.empty([FLAGS.BUFFER_SIZE])
        self.next_eqrt_obs_que = np.empty([FLAGS.BUFFER_SIZE, self.N_EQRT, self.N_FEATURES])
        self.next_ivrt_obs_que = np.empty([FLAGS.BUFFER_SIZE, self.N_IVRT, self.N_FEATURES])

        # Temporal storage until (s,a,r,s') is stored in the queue
        self.eqrt_obs_temp = np.empty([self.N_EQRT, self.N_FEATURES])
        self.ivrt_obs_temp = np.empty([self.N_IVRT, self.N_FEATURES])
        self.act_temp = np.empty([self.OUT_DIM])
        self.rew_temp = np.empty([1])
        self.next_eqrt_obs_temp = np.empty([self.N_EQRT, self.N_FEATURES])
        self.next_ivrt_obs_temp = np.empty([self.N_IVRT, self.N_FEATURES])

    def q_network(self, params):
        """
        Creating Q network as tensor graph
        Return placeholder for input obs
        Return Q value for observed state obs
        """
        # Input state observations
        obs = tf.placeholder(tf.float32, [None, self.OBS_DIM])

        # Extract weights and biases from params
        W, b = params[0], params[1]

        # Hidden matrix of units in hidden layers
        h = [None] * self.LAYER_SIZE.__len__()

        # Consider input states as hidden matrix in 0-th layer
        h[0] = obs

        # Fully-connected network
        for i in range(self.LAYER_SIZE.__len__() - 1):
            h_matmul = tf.matmul(h[i], W[i]) + b[i]
            h[i+1] = tf.nn.leaky_relu(h_matmul, alpha=FLAGS.REL_ALP)

        Q = tf.squeeze(h[-1])

        return obs, Q

    def sample_action(self, Q, obs, eps):
        """
        Uniformly sample action or get optimal action from main network
        Return action in one-hot format
        """
        # Get action index
        if self.test_mode == False:
            action_index = np.random.randint(self.OUT_DIM)
            if self.global_step > FLAGS.BUFFER_SIZE:
                # Calculate act_values here to use in main.py
                self.act_values = self.sess.run(Q, feed_dict={self.obs: obs})
                if np.random.random() > eps:
                    action_index = np.argmax(self.act_values)  # current optimal action
        else:
            self.act_values = self.sess.run(Q, feed_dict={self.obs: obs})
            action_index = np.argmax(self.act_values)  # optimal action from global network

        # Convert action to one-hot vector
        action = np.zeros(self.OUT_DIM)
        action[action_index] = 1

        return action

    def update_buffer(self, eqrt_s, ivrt_s, a, r, eqrt_sn, ivrt_sn):
        """
        Update replay buffer
        """
        self.eqrt_obs_que[self.exp_pointer] = eqrt_s
        self.ivrt_obs_que[self.exp_pointer] = ivrt_s
        self.act_que[self.exp_pointer] = a
        self.rew_que[self.exp_pointer] = r
        self.next_eqrt_obs_que[self.exp_pointer] = eqrt_sn
        self.next_ivrt_obs_que[self.exp_pointer] = ivrt_sn

        # Update pointer
        self.exp_pointer = self.exp_pointer + 1
        if self.exp_pointer == FLAGS.BUFFER_SIZE:
            self.exp_pointer = 0

    def act(self, obs):
        """
        Return ID of selected item as action to environment (used in main.py)
        Update replay buffer
        """

        # Extract and concatenate obs received from environment
        in_obs = np.concatenate((obs['invariant_array'], obs['equivariant_array']), axis=0)
        in_obs = in_obs.reshape(-1, self.OBS_DIM)

        if self.test_mode == False:
            self.eqrt_obs_temp = cp.deepcopy(self.next_eqrt_obs_temp)
            self.next_eqrt_obs_temp = cp.deepcopy(obs['equivariant_array'])

            self.ivrt_obs_temp = cp.deepcopy(self.next_ivrt_obs_temp)
            self.next_ivrt_obs_temp = cp.deepcopy(obs['invariant_array'])

            # Update replay buffer
            self.update_buffer(self.eqrt_obs_temp, self.ivrt_obs_temp, self.act_temp,
                self.rew_temp, self.next_eqrt_obs_temp, self.next_ivrt_obs_temp)

            # Reshape the obs matrix and get action
            action = self.sample_action(self.Q1, in_obs, self.eps)

            self.act_temp = cp.deepcopy(action)
        else:
            action = self.sample_action(self.Q1, in_obs, -1.)

        recommended_item_id = np.argmax(action)

        return [recommended_item_id]

    def enable_test_mode(self):
        """Enable test mode
        """
        self.test_mode = True

    def disable_test_mode(self):
        """Disable test mode
        """
        self.test_mode = False

    def receive_reward(self, reward):
        """
        Get reward from main.py
        Perform learning on main networks
        Apply gradients to global parameters
        Update parameters of local networks
        """
        self.rew_temp = reward
        if self.global_step > FLAGS.BUFFER_SIZE:
            # Decrease epsilon
            if self.eps > 0:
                self.eps = np.max([self.eps * self.EPS_DIS_CONST, self.FINAL_EPS])

            if self.test_mode == False:
                self.sess.run([self.train_step, self.apply_grads], feed_dict=self.feed_batch())
                # Update target network
                if self.global_step % FLAGS.UPDATE_TARGET == 0:
                    self.copy_parameters()

    def copy_parameters(self):
        """
        Copy global parameters to local main and target networks
        """
        self.sess.run(self.copy_ops)
        # self.sess.run(self.copy_to_main_ops)
        # self.sess.run(self.copy_to_target_ops)

    def feed_batch(self, batch_size=FLAGS.BATCH_SIZE):
        """
        Return batch_size * (obs, action, reward, next_obs)
        (in the form of dictionary)
        """
        feed = {}
        if self.global_step >= FLAGS.BUFFER_SIZE:
            rand_indexs = np.random.choice(FLAGS.BUFFER_SIZE, batch_size, False)

            # Randomly sample experience from replay buffer
            eqrt_obs_mat = self.eqrt_obs_que[rand_indexs]
            ivrt_obs_mat = self.ivrt_obs_que[rand_indexs]
            act_mat = self.act_que[rand_indexs]
            next_eqrt_obs_mat = self.next_eqrt_obs_que[rand_indexs]
            next_ivrt_obs_mat = self.next_ivrt_obs_que[rand_indexs]

            obs_mat = np.concatenate((ivrt_obs_mat, eqrt_obs_mat), axis=1)
            next_obs_mat = np.concatenate((next_ivrt_obs_mat, next_eqrt_obs_mat), axis=1)

            # Update feed dict
            feed.update({self.obs: np.reshape(obs_mat, (batch_size, self.OBS_DIM))})
            feed.update({self.action: act_mat})
            feed.update({self.reward: self.rew_que[rand_indexs]})
            feed.update({self.next_obs: np.reshape(next_obs_mat, (batch_size, self.OBS_DIM))})

        return feed

    def set_step(self, step=0):
        """
        Set the self.global_step as step
        """
        self.global_step = step

    def get_step(self):
        """
        Return the current global step self.global_step
        """
        return self.global_step

    def increase_step(self):
        """
        Increase the self.global_step
        """
        self.global_step += 1

    def get_main_params(self):
        """
        Return parameters of main network
        """
        return self.sess.run(self.main_params)

    def get_targ_params(self):
        """
        Return parameters of target network
        """
        return self.sess.run(self.targ_params)

    def get_loss(self):
        """
        return current loss
        """
        # return self.sess.run(self.loss, feed_dict=self.feed_batch())
        return -1. * np.ones(self.N_CHOOSE)

    def get_q_value(self):
        """return q-value
        """
        # q_val = self.sess.run(self.Q1, feed_dict=self.feed_batch())
        # return q_val * np.ones(self.N_CHOOSE)
        return -1. * np.ones(self.N_CHOOSE)
