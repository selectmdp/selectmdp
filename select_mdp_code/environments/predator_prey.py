import numpy as np
import config

FLAGS = config.flags.FLAGS

class Environment:
    def __init__(self, n_eqrt=FLAGS.N_EQUIVARIANT, n_ivrt=FLAGS.N_INVARIANT, train_mode=True):
        self.N_EQRT = n_eqrt        # Number of predators
        self.N_IVRT = n_ivrt        # Number of preys
        self.LB = -FLAGS.MAP_BND    # Lower bound of map
        self.UB = FLAGS.MAP_BND     # Upper bound of map
        self.N_FEATURES = FLAGS.N_FEATURES
        self.N_CHOOSE = FLAGS.N_CHOOSE

        self.LB_MV_NOISE = -FLAGS.STEP_SIZE * FLAGS.NOISE_RATIO
        self.UB_MV_NOISE = FLAGS.STEP_SIZE * FLAGS.NOISE_RATIO

        # self.DIST_SCALE = 2/np.sqrt(self.N_EQRT)
        self.DIST_SCALE =  0.15
        self.DEAD_DIST_SINGLE = self.DIST_SCALE * FLAGS.DEAD_DIST_SINGLE
        self.DEAD_DIST_DOUBLE = self.DIST_SCALE * FLAGS.DEAD_DIST_DOUBLE

        # self.DEAD_DIST_SINGLE = np.clip(self.DEAD_DIST_SINGLE,
        #     a_min=FLAGS.LB_DEAD_DIST_SINGLE, a_max=self.DEAD_DIST_SINGLE)
        # self.DEAD_DIST_DOUBLE = np.clip(self.DEAD_DIST_DOUBLE,
        #     a_min=FLAGS.UB_DEAD_DIST_DOUBLE, a_max=self.DEAD_DIST_DOUBLE)

        self.reset()

    def reset(self):
        """Reset positions of predators and preys
        """
        self.eqrt_state = np.random.uniform(self.LB, self.UB,
            size=[self.N_EQRT, self.N_FEATURES])
        self.ivrt_state = np.random.uniform(self.LB, self.UB,
            size=[self.N_IVRT, self.N_FEATURES])

        if FLAGS.SORTED:
            self.sort_state()

    def step(self, action):
        """Receive action, update state, return reward
        Received action values meaning:
            -1: unselected, 0: stay, 1: left, 2: up, 3: right, 4: down
        """
        # Get indexes of selected predators
        selected_pred_idx = np.nonzero(action!=-1)[0]

        # Get moves of predators based on received actions
        eqrt_move = np.zeros((self.N_EQRT, self.N_FEATURES))
        eqrt_move[np.nonzero(action==1)[0],0] = -FLAGS.STEP_SIZE    # left
        eqrt_move[np.nonzero(action==2)[0],1] = FLAGS.STEP_SIZE     # up
        eqrt_move[np.nonzero(action==3)[0],0] = FLAGS.STEP_SIZE     # right
        eqrt_move[np.nonzero(action==4)[0],1] = -FLAGS.STEP_SIZE    # down

        # Randomly generate moves of preys
        ivrt_move = np.zeros((self.N_IVRT, self.N_FEATURES))
        for i in range(self.N_IVRT):
            along_x = np.random.randint(self.N_FEATURES)    # along x or y axis
            direction = np.random.choice([-1, 0, 1])        # left/down (-1) or stay (0) or right/up (1)
            ivrt_move[i, along_x] = direction * FLAGS.STEP_SIZE

        # Noise for movements of predators and preys
        eqrt_noise = np.random.uniform(self.LB_MV_NOISE, self.UB_MV_NOISE,
            size=[self.N_EQRT, self.N_FEATURES])
        ivrt_noise = np.random.uniform(self.LB_MV_NOISE, self.UB_MV_NOISE,
            size=[self.N_IVRT, self.N_FEATURES])

        # Add noise to moves of preys
        eqrt_move += eqrt_noise
        ivrt_move += ivrt_noise

        # Scale the movements
        eqrt_move *= self.DIST_SCALE
        ivrt_move *= self.DIST_SCALE

        # Update current state
        self.eqrt_state += eqrt_move
        self.ivrt_state += ivrt_move

        # Make predators and preys stay inside the map
        self.eqrt_state = np.clip(self.eqrt_state, self.LB, self.UB)
        self.ivrt_state = np.clip(self.ivrt_state, self.LB, self.UB)

        # Initialize reward
        reward = FLAGS.INIT_REWARD

        # Penalty because of going outside the map
        reward -= np.count_nonzero(self.eqrt_state[selected_pred_idx,:] == self.LB) * FLAGS.PENALTY_OUT
        reward -= np.count_nonzero(self.eqrt_state[selected_pred_idx,:] == self.UB) * FLAGS.PENALTY_OUT

        # Check whether preys are caught
        #TODO: more vectorization?
        for i in range(len(self.ivrt_state)):
            # dist = np.linalg.norm(self.ivrt_state[i]-self.eqrt_state, axis=1)
            dist = np.linalg.norm(self.ivrt_state[i]-self.eqrt_state[selected_pred_idx,:], axis=1)
            # print('dis', dist)
            if np.count_nonzero(dist < self.DEAD_DIST_DOUBLE) == 2:
                reward += FLAGS.REWARD_DOUBLE
                self.ivrt_state[i] = np.random.uniform(self.LB, self.UB, size=[self.N_FEATURES])
            elif np.min(dist) < self.DEAD_DIST_SINGLE:
                reward += FLAGS.REWARD_SINGLE
                self.ivrt_state[i] = np.random.uniform(self.LB, self.UB, size=[self.N_FEATURES])

        if FLAGS.SORTED:
            self.sort_state()

        return reward/self.N_CHOOSE

    def get_state(self):
        """Return current state in dictionary format
        i.e. {'eqrt': numpy array, 'ivrt': numpy array}
        """
        state_dict = {}

        state_dict['equivariant_array'] = self.eqrt_state
        state_dict['invariant_array'] = self.ivrt_state

        return state_dict

    def sort_state(self):
        """Sort states by x coordinate in descending order
        """
        self.eqrt_state = sorted(self.eqrt_state,
            key=lambda x: x[0], reverse=True)
        self.eqrt_state = np.array(self.eqrt_state)
        self.ivrt_state = sorted(self.ivrt_state,
            key=lambda x: x[0], reverse=True)
        self.ivrt_state = np.array(self.ivrt_state)
