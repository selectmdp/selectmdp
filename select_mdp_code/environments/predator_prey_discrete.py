import numpy as np
import config
import random
import copy as cp

FLAGS = config.flags.FLAGS

class Environment:
    def __init__(self, n_eqrt=FLAGS.N_EQUIVARIANT, n_ivrt=FLAGS.N_INVARIANT, train_mode=True):
        self.N_EQRT = n_eqrt        # Number of predators
        self.N_IVRT = n_ivrt        # Number of preys
        self.N_GRID = FLAGS.GRID_SIZE
        self.N_GRID_1 = self.N_GRID - 1
        self.N_SUBACT = FLAGS.N_SUBACT


        self.LB = -FLAGS.MAP_BND    # Lower bound of map #TODO
        self.UB = FLAGS.MAP_BND     # Upper bound of map #TODO

        self.N_FEATURES = FLAGS.N_FEATURES
        self.N_CHOOSE = FLAGS.N_CHOOSE

        # self.LB_MV_NOISE = -FLAGS.STEP_SIZE * FLAGS.NOISE_RATIO
        # self.UB_MV_NOISE = FLAGS.STEP_SIZE * FLAGS.NOISE_RATIO

        # self.DIST_SCALE = 2/np.sqrt(self.N_EQRT)
        self.DIST_SCALE =  1/float(self.N_GRID_1) #TODO

        self.DEAD_DIST = 1.5
        self.reset()

    def reset(self, test_check=False):
        """Reset positions of predators and preys
        """
        # self.matrix_GRID = constant matrix for all grid points
        self.matrix_GRID = np.ones((self.N_GRID*self.N_GRID, 2))
        for i in range(self.N_GRID * self.N_GRID):
            self.matrix_GRID[i] = np.array([int(i/self.N_GRID), i%self.N_GRID])
        
        # self.state_grid = moving matrix for states: contain 0~N-1 predator and prey both
        self.state_grid = self.matrix_GRID[np.random.choice(range(len(self.matrix_GRID)), (self.N_EQRT+self.N_IVRT))] 

        # set reward differently when test mode
        self.set_minus_reward()
        if test_check:
            self.set_zero_reward()

        # if FLAGS.SORTED:
        #     self.sort_state()

    def set_minus_reward(self):
        """
        Set reward as minus for better exploration
        """
        self.INIT_REWARD= FLAGS.INIT_REWARD
        self.REWARD_COLLIDE = FLAGS.REWARD_COLLIDE
        pass

    def set_zero_reward(self):
        """Define the reward function for test (set the zero reward for collide)
        """
        self.INIT_REWARD= 0
        self.REWARD_COLLIDE = 0
        pass
        


    def step(self, action):
        """Receive action, update state, return reward
        Received action values meaning:
            -1: unselected, 0: stay, 1: left, 2: up, 3: right, 4: down
        """
        # Get indexes of selected predators
        action = np.transpose(action)
        np.random.shuffle(action)
        action = np.transpose(action)
        self.reward =  self.INIT_REWARD # reward

        # Randomly generate moves of preys
        #TODO: vectorization?
      
        for i in range(self.N_IVRT):
            self.move_agent(-i-1)
        
        for i in range(self.N_CHOOSE):
            # print(' action[0][i], action[1][i]',  action[0][i], action[1][i])
            self.move_agent(action[0][i], action[1][i])
 
        # Initialize reward
        self.reward = self.INIT_REWARD

        # Check whether preys are caught
        #TODO: more vectorization?
        ivrt_state = self.state_grid[self.N_EQRT:]
        eqrt_state = self.state_grid[action[0]]
        # print('before regeneration', ivrt_state, eqrt_state)
        for i in range(self.N_IVRT):
            # dist = np.linalg.norm(self.ivrt_state[i]-self.eqrt_state, axis=1)
            dist = np.linalg.norm(ivrt_state[i]-eqrt_state, axis=1)
            # print('dis', dist)
            if np.count_nonzero(dist < 1.5) > 1:
                self.reward += FLAGS.REWARD_DOUBLE
                # print('caught prey', ivrt_state[i])
                self.regenerate_prey(self.N_EQRT+i)
            # elif np.min(dist) < 1.5:
            #     self.reward += FLAGS.REWARD_SINGLE
            #     # print('caught prey', ivrt_state[i])
            #     self.regenerate_prey(self.N_EQRT+i) 

        # print('reward', self.reward)
        return self.reward/float(self.N_CHOOSE)

    def move_agent(self, agent_num, command=-1):
        """Move the agent with the consideration of the collision. If the agent collides to existence of the agent, it stops and get the lower reward
        -------
        agent_num: number of the agent to be moved
        command: command for moving the agent
        """
        if command == -1: # prey agent
            command = np.random.randint(0, 2*self.N_SUBACT)
            if command >=self.N_SUBACT:
                command = 0
   
        original_pos = cp.deepcopy(self.state_grid[agent_num])
        # print('before', self.state_grid[agent_num])
        # print('command', command)
       

        if command == 0: # stay
            self.state_grid[agent_num] = original_pos
        elif command == 1: # left
            self.state_grid[agent_num] = original_pos + np.array([-1,0])
        elif command == 2: # up
            self.state_grid[agent_num] = original_pos + np.array([0,1])
        elif command == 3: # right
            self.state_grid[agent_num] = original_pos + np.array([1,0])
        elif command == 4: # below
            self.state_grid[agent_num] = original_pos + np.array([0,-1])
        else:
            print('prey predators command has wrong format')
            exit(0)
      
        # print('before collision', self.state_grid[agent_num])

        self.check_boundary_collision(agent_num,  original_pos)
        # print('after', self.state_grid[agent_num])


        return None


    def check_boundary_collision(self, agent_num, original_pos):
        # TODO: check boundary of grid, negative reward 
        if self.check_boundary(agent_num):
            # print('check_boundary')
            self.state_grid[agent_num] = original_pos
            self.reward += self.REWARD_COLLIDE

        # check collision
        elif not self.check_collision(agent_num):
            # print('check collision')
            self.state_grid[agent_num] = original_pos
            self.reward += self.REWARD_COLLIDE

        return None

    def check_boundary(self, agent_num):
        """Check the collision between the agent with agent_num with the grid boundary.
        ----
        return true is really true.
        """
        boundary = False
        if self.state_grid[agent_num][0]==-1 or self.state_grid[agent_num][0] == self.N_GRID or self.state_grid[agent_num][1]==-1 or self.state_grid[agent_num][1] == self.N_GRID:
            boundary = True

        return boundary

    def check_collision(self, agent_num):
        """Check the collision betweeen the agents with number agent_num and others
        Return True if collision occurs
        """
        # print('delete_array', np.delete(self.state_grid, agent_num, 0) - self.state_grid[agent_num])
        delete = np.delete(self.state_grid, agent_num, 0) - self.state_grid[agent_num]
        count_nonzero = np.count_nonzero(delete, 1)
        # print('count_nonzero',count_nonzero)
        collision = np.all(count_nonzero)
        return collision

    def regenerate_prey(self, agent_num):
        """Regenerate the caught preys.
        """
        grid_rows = set(map(tuple, self.matrix_GRID))
        agent_rows = set(map(tuple, self.state_grid))

        # print('grid_rows.difference(agent_rows)',grid_rows.difference(agent_rows))
        self.state_grid[agent_num] = np.array(random.sample(grid_rows.difference(agent_rows), 1))
        # print('regenerate', self.state_grid[agent_num])

        return None


    def get_state(self):
        """Return current state in dictionary format
        i.e. {'eqrt': numpy array, 'ivrt': numpy array}
        """
        state_dict = {}
        state_dict['equivariant_array'] = 2/float(self.N_GRID_1) * self.state_grid[0:self.N_EQRT] - 1
        state_dict['invariant_array'] = 2/float(self.N_GRID_1) * self.state_grid[self.N_EQRT:] -1
        # print('state_grid', self.state_grid)
        #  2/float(self.N_GRID_1) *

        # print('state_dict', state_dict)
        return state_dict


