import tensorflow as tf

flags = tf.flags

# Hot parameters ------------
N = 4 # unselectable
M = 50  # selectable
K = 6 # Number of actions
F = 2  # Number of element features
SUB = 5  # Numb of sub actions

TRAIN_STEP = 250
TRAIN_EPISODE = 8
BUFFER_SIZE = 200
# ---------------------------

# Greedy
flags.DEFINE_float("BIG_BAD_SIZE", 0.2, "Big bad circle threshold")

## Environment
# flags.DEFINE_string("ENVIRONMENT", "circle_env_good", "Environment to use")  # circle_env | predator_prey

flags.DEFINE_string("ENVIRONMENT", "predator_prey_discrete", "Environment to use")  # circle_env | predator_prey

flags.DEFINE_integer("N_INPUT", 10, "Number of equivariant items to be reloaded")


flags.DEFINE_integer("N_EQUIVARIANT", M, "Number of equivariant elements")
flags.DEFINE_integer("N_INVARIANT", N, "Number of invariant elements")
flags.DEFINE_integer("N_FEATURES", F, "Number of features of elements")
flags.DEFINE_integer("N_CHOOSE", K, "Number of chosen element")
flags.DEFINE_integer("N_SUBACT", SUB, "Number of subactions for each agents")

flags.DEFINE_integer("SORTED", 0, "0: Env returns normal state, 1: Env returns sorted state, 2: shuffle")
# flags.DEFINE_integer("SHUFFLED", 0, "0: Env returns normal state, 1: Env returns shufled state")


# Circles
flags.DEFINE_float("MAX_RADIUS", 0.45, "Upper bound of circle radius")
flags.DEFINE_float("LB_CENTER", -0.5, "Lower bound of circle center coordinate")
flags.DEFINE_float("UB_CENTER", 0.5, "Upper bound of circle center coordinate")
flags.DEFINE_float("MAX_NOISE", 0.01, "Upper bound of noise")
flags.DEFINE_float("EXPAND_CONST", 0.1, "How much the bad circle will expand at each step")
flags.DEFINE_float("INIT_RADIUS", 0.01, "the initial radius for invariant circles")
flags.DEFINE_float("ACTION_MOVE_DIST", 0.1, "moving size by actions")

flags.DEFINE_float("GPU_OPTION", 0.43, "gpu options")

# Predator-prey environment
flags.DEFINE_integer("MAP_BND", 1, "Boundary of play map")
flags.DEFINE_float("STEP_SIZE", 1, "Size of one step of moving")
flags.DEFINE_float("DEAD_DIST_SINGLE", 1, "Distance within which a prey is caught by 1 predator")
flags.DEFINE_float("LB_DEAD_DIST_SINGLE", 0.25, "Lower bound for scaled value of DEAD_DIST_SINGLE")
flags.DEFINE_float("DEAD_DIST_DOUBLE", 1, "Distance within which a prey is caught by 2 predators")
flags.DEFINE_float("UB_DEAD_DIST_DOUBLE", 0.35, "Lower bound for scaled value of DEAD_DIST_DOUBLE")
flags.DEFINE_float("INIT_REWARD", -0.005, "Initial negative reward")
flags.DEFINE_float("REWARD_COLLIDE", -0.005, "negative reward for the collision of agents")
flags.DEFINE_float("REWARD_SINGLE", 0.00, "Given reward when a prey is caught by a predator")
flags.DEFINE_float("REWARD_DOUBLE", 1, "Given reward when a prey is caught by 2 predators")
flags.DEFINE_float("PENALTY_OUT", 0.02, "Penalty scale of going outside the map")
flags.DEFINE_float("NOISE_RATIO", 0.1, "Noise ratio for moving distance of predators and preys")
flags.DEFINE_integer("GRID_SIZE", 4, "Size of the Grid")


## Agent 
# flags.DEFINE_string("AGENT", "dummy", "choose action uniformly randomly")   # dummy | dqn | greedy
# flags.DEFINE_string("AGENT", "greedy", "choose action uniformly randomly")   # dummy | dqn | greedy
# flags.DEFINE_string("AGENT", "greedy_good", "choose action uniformly randomly")   # dummy | dqn | greedy

# flags.DEFINE_string("AGENT", "dqn", "choose action uniformly randomly")   # dummy | dqn | greedy
# flags.DEFINE_string("AGENT", "dqn_myopic", "choose action uniformly randomly")   # dummy | dqn | greedy
flags.DEFINE_string("AGENT", "CENTRAL_GREEDY", "choose action uniformly randomly")   # dummy | dqn | greedy
# flags.DEFINE_string("AGENT", "dqn_ind", "choose action uniformly randomly")   # dummy | dqn | greedy
# flags.DEFINE_string("AGENT", "dqn_individual", "choose action uniformly randomly")   # dummy | dqn | greedy




## Neural Network Architectures
# flags.DEFINE_string("NETWORK", "NONE", "Not use any network, even dummy can update NN")
# flags.DEFINE_string("NETWORK", "VANILLA", "Network to use")
# flags.DEFINE_string("NETWORK", "DIFFEINET", "Network to use") # general EINET: do not share the parameters among K networks
# flags.DEFINE_string("NETWORK", "SHAREEINET", "Network to use") # normal EINET: share parameters among K networks
flags.DEFINE_string("NETWORK", "PROGRESSIVE", "Network to use") # Progressive Networks
# flags.DEFINE_string("NETWORK", "PROGRESSIVE_1_K", "Network to use") # Progressive Networks
# flags.DEFINE_string("NETWORK", "PROGRESSIVE_ROOT", "Network to use") # Progressive Networks




## Network Structure
flags.DEFINE_integer("NWRK_EXPAND", 6, "Expansion ratio of hidden units in neural network")
flags.DEFINE_integer("LAYERS", 4, "Number of layers in the network")
flags.DEFINE_float("RATIO_PROGRESSIVE", 0.5, "Ratio of the learning small networks")

# flags.DEFINE_integer("EQ_SIZE1", 32, "1st hidden equivariant node size")
# flags.DEFINE_integer("EQ_SIZE2", 32, "2nd hidden equivariant node size")
# flags.DEFINE_integer("EQ_SIZE3", 32, "3rd hidden equivariant node size")
# flags.DEFINE_integer("EQ_SIZE4", 32, "4th hidden equivariant node size")


## Deep Learning hyperparameters 
flags.DEFINE_float("LEARNING_RATE", 0.001, "Learning rate")
flags.DEFINE_float("ADAM_EP", 0.01, "epsilon for ADAM optimizer")
flags.DEFINE_float("REL_ALP", 0.1, "alpha for leaky relu")
flags.DEFINE_float("DROPOUT", 0.6, "dropout rates")
flags.DEFINE_integer("XAVIER", 1, "Initialization type, 0: 1/sqrt(1), 1: 1/sqrt(N), 2: 1/N")

## RL hyperparamters
EPS_DIS_CONST = (TRAIN_STEP * TRAIN_EPISODE - BUFFER_SIZE)/float(8)
# Reward
flags.DEFINE_float("GAMMA", 0.99, "Discount factor")
# Exploration
flags.DEFINE_float("INIT_EPS", 1, "initial parameter of EPS")
flags.DEFINE_float("FINAL_EPS", 0.1, "final parameter of EPS")
# Replay buffer
flags.DEFINE_float("EPS_DIS_CONST", EPS_DIS_CONST, "exploration decreasing rate")
flags.DEFINE_integer("BATCH_SIZE", 3, "batch size which using in buffer")
flags.DEFINE_integer("BUFFER_SIZE", BUFFER_SIZE, "buffer queue size")  # IMPORTANT
# Target network Update Period
flags.DEFINE_integer("UPDATE_TARGET", 200, "Update period of the target network")


## Training configurations
flags.DEFINE_integer("TRAIN_STEP", TRAIN_STEP, "Number of the steps in a episode")
flags.DEFINE_integer("TRAIN_EPISODE", TRAIN_EPISODE, "Number of training episodes")
flags.DEFINE_integer("SAVE_PERIOD", 2500, "Saving intermediate result every SAVE_PERIOD steps")
flags.DEFINE_integer("SAVE_REPEAT", 10, "Saving intermediate result during SAVE_REPEAT steps")
flags.DEFINE_integer("MAX_TO_KEEP", 10, "Saving how mnay tensor parameters")
flags.DEFINE_integer("REDUCED_TENSOR", 1, "0: save all informations, 1: only reward_train, q_test, reward")

## Save and Reload parameters
flags.DEFINE_integer("RELOAD_EP", 0, "  number of the episode that reloaded")
flags.DEFINE_integer("TOTAL_RELOAD", 10, "number of divides training")


# Simulation
flags.DEFINE_integer("MODE", 0, "0: train, 1: test")
flags.DEFINE_integer("SET_SEED", 1, "1: seed exists, 0: not exists")
flags.DEFINE_integer("SEED", 0, "random seed")


## TEST SETTINGS
# The following configurations are used to locate trained parameters
flags.DEFINE_string("TEST_AGENT", "enn", "Agent to test")   # dummy | greedy | dqn | dqn-da | dqn-sort | enn
flags.DEFINE_integer("TRAINED_SEED", 0, "Seed value used in training")
flags.DEFINE_string("TRAINED_N_QUERY", "10", "Numbers of queries used in training")
flags.DEFINE_float("TRAINED_LR", 0.003, "Learning rate used in training")
flags.DEFINE_integer("TRAINED_BUF_SIZE", 200000, "Buffer size used in training")
flags.DEFINE_string("TRAINED_QUERIES_SYNC_UPDATE", "10-50-100-500", "Number of queries used in training with synchronous update")

# Testing configurations
flags.DEFINE_integer("N_QUERY_TEST", 100, "Number of queries to test")
flags.DEFINE_integer("TEST_EPISODE_REAL", 100, "Number of episodes for testing with real environment")
flags.DEFINE_integer("TEST_EPISODE_SYNTH", 1, "Number of episodes for testing with multi-markov environment")
flags.DEFINE_integer("TEST_EPISODE", 1, "Temporary number of testing episodes")
flags.DEFINE_integer("TEST_STEP", 100000, "Number of steps to run in test")
