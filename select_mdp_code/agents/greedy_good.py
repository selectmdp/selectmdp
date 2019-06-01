# greedy agent
# have to mention in paper that exhaustive trade-off calculation is computationally nasty
import random
import config
import tensorflow as tf
import environments.circle_utils as cu
import numpy as np
import copy as cp

FLAGS = tf.flags.FLAGS

class AgentCircle(object):

    def __init__(self, scope, session = None, feature_size=None, main_params=None, target_params=None):

        self.N_EQRT = FLAGS.N_EQUIVARIANT
        self.N_IVRT = FLAGS.N_INVARIANT
        self.N_CHOOSE = FLAGS.N_CHOOSE

    def act(self, obs, train=True):
        """
        obs is a dictionary

        {'equivariant': numpy array, 'invariant': numpy array}

        numpy array = [[good_circle.x, good_circle.y, good_circle.r], ...]

        """

        action_list = []
        subaction_list = []

        k = FLAGS.N_CHOOSE

        state_dict = obs

        # Never modify these variables
        g_array = state_dict['equivariant_array']
        b_array = state_dict['invariant_array']

        # Indices of good and bad circles. Never modify.
        g_list = range(0, g_array.shape[0])
        b_list = range(0, b_array.shape[0])

        # Modify these
        # selectable --> sable
        # sable_coll_list, _ = cu.check_collision(g_array, b_array)
        # sable_uncoll_list = [i for i in g_list if i not in sable_coll_list]
        remained_index = [i for i in g_list]

        # sable_coll_array = g_array[sable_coll_list]
        # remained_matrix = g_array[remained_index]



        # print("g_list: {}".format(g_list))
        # print("b_list: {}".format(b_list))

        # print("sable_coll_list: {}".format(sable_coll_list))
        # print("sable_uncoll_list: {}".format(sable_uncoll_list))

        # print("GREEDY START")

        stop_zero = False
        stop_one = False
        n_select = 0
        choosed_matrix = np.zeros([0,FLAGS.N_FEATURES])

        # safety zone for radius
        Safe_Dist = 3 * 0.5 * FLAGS.MAX_NOISE + 0.55 *0.6 * FLAGS.MAX_RADIUS * FLAGS.N_CHOOSE/np.float(FLAGS.N_EQUIVARIANT)
        
        command_list=np.arange(FLAGS.N_SUBACT)

        for cand_index in range(FLAGS.N_EQUIVARIANT):
            np.random.shuffle(command_list)
            # print('command', command_list)
            for command in command_list:
                # move with the actions
                cand_pos = cp.deepcopy(g_array[cand_index])
                cand_pos[-1]+=Safe_Dist
                #TODO: print _list
                # print('before remained_index',g_array[cand_index])
                # print('before cand_pos',cand_pos)
                
                if command==1:
                    cand_pos[0] -= FLAGS.ACTION_MOVE_DIST #left
                elif command==2:
                    cand_pos[1]+= FLAGS.ACTION_MOVE_DIST #up
                elif command==3:
                    cand_pos[0]+= FLAGS.ACTION_MOVE_DIST #right
                elif command==4:
                    cand_pos[1]-= FLAGS.ACTION_MOVE_DIST #down
                
                #TODO: print list
                # print('cand_pos',cand_pos)
                cand_pos = np.reshape(cand_pos, [1,-1])

                # check collision if more than 1 circle selected
    
                # print('cand_pos',cand_pos)
                # print('choosed matrx', choosed_matrix)
                if n_select==0:
                    action_list.append(cand_index)
                    subaction_list.append(command)
                    choosed_matrix = np.append(choosed_matrix, cand_pos)
                    n_select+=1
                    
                    choosed_matrix = np.reshape(choosed_matrix, [-1,FLAGS.N_FEATURES])
                    break
                else:
                    if len(cu.check_collision(cand_pos, choosed_matrix)[1])==0:
                        action_list.append(cand_index)
                        subaction_list.append(command)
                        choosed_matrix = np.append(choosed_matrix, cand_pos)
                        n_select+=1
                        choosed_matrix = np.reshape(choosed_matrix, [-1,FLAGS.N_FEATURES])
                    break

                
            if n_select==FLAGS.N_CHOOSE:
                break

        # when there is no selection is able
        if n_select<FLAGS.N_CHOOSE:
            remained_set = set(np.arange(FLAGS.N_EQUIVARIANT))-set(action_list)
            action_list += random.sample(list(remained_set), k=FLAGS.N_CHOOSE-n_select)
            subaction_list += list(np.random.random_integers(0, FLAGS.N_SUBACT-1, FLAGS.N_CHOOSE-n_select))
            

        # print("")
        # print("ACTION")
        # print("action_list: {}".format(action_list))
        # print("")
        actions = []
        actions.append(action_list)
        actions.append(subaction_list)

        return np.array(actions)

    def receive_next_state(self, obs, train=True):
        return None

    def receive_reward(self, reward):
        pass

    def increase_step(self):
        pass

    def get_loss(self):
        return np.zeros(FLAGS.N_CHOOSE)

    def get_q_value(self):
        return np.zeros(FLAGS.N_CHOOSE)

    def global_step(self):
        pass

    def copy_parameters(self):
        pass

    def save_network(self):
        pass

class AgentPredPrey:

    def __init__(self, scope, session = None, feature_size=None, main_params=None, target_params=None):

        self.N_EQRT = FLAGS.N_EQUIVARIANT
        self.N_IVRT = FLAGS.N_INVARIANT
        self.N_CHOOSE = FLAGS.N_CHOOSE

        self.DIST_SCALE = 0.3
        # self.DIST_SCALE = 1/ np.sqrt(self.N_EQRT + self.N_IVRT)

        self.DEAD_DIST_SINGLE = self.DIST_SCALE * FLAGS.DEAD_DIST_SINGLE
        # self.DEAD_DIST_SINGLE = np.clip(self.DEAD_DIST_SINGLE,
        #     a_min=FLAGS.LB_DEAD_DIST_SINGLE, a_max=self.DEAD_DIST_SINGLE)

    def act(self, obs, train=True):
        preds = obs['equivariant_array']
        preys = obs['invariant_array']

        # For each predator, calculate the smallest distance to the nearest prey
        smallest_distance = np.zeros(self.N_EQRT)
        nearest_prey_idx = np.zeros(self.N_EQRT, dtype=int)
        for i in range(self.N_EQRT):
            dist = np.linalg.norm(preds[i]-preys, axis=1)
            smallest_distance[i] = np.min(dist)
            nearest_prey_idx[i] = np.argmin(dist)

        # Select k predators by choosing k smallest distances
        idx = np.argpartition(smallest_distance, self.N_CHOOSE)
        k_selected_preds_idx = idx[:self.N_CHOOSE]
        k_smallest_distance = smallest_distance[k_selected_preds_idx]
        k_nearest_preys_idx = nearest_prey_idx[k_selected_preds_idx]

        # Select greedy actions
        action_list = np.zeros(self.N_EQRT) - 1
        for i in range(len(k_smallest_distance)):
            pred_idx = k_selected_preds_idx[i]
            # if k_smallest_distance[i] < self.DEAD_DIST_SINGLE:
            #     # action_list[pred_idx] = 0    # stay
            #     pass
            # else:
            prey_idx = k_nearest_preys_idx[i]
            coordinate_diff = preds[pred_idx] - preys[prey_idx]
            if np.abs(coordinate_diff[0]) > np.abs(coordinate_diff[1]): # Move along x axis
                if coordinate_diff[0] > 0:  # x_pred > x_prey
                    action_list[pred_idx] = 1   # left
                else:   # x_pred <= x_prey
                    action_list[pred_idx] = 3   # right
            else:   # Move along y axis
                if coordinate_diff[1] > 0:  # y_pred > y_prey
                    action_list[pred_idx] = 4   # down
                else:   # y_pred <= y_prey
                    action_list[pred_idx] = 2   # up

        return action_list

    def receive_next_state(self, obs, train=True):
        return None

    def receive_reward(self, reward):
        pass

    def increase_step(self):
        pass

    def get_loss(self):
        return np.zeros(FLAGS.N_CHOOSE)

    def get_q_value(self):
        return np.zeros(FLAGS.N_CHOOSE)

    def global_step(self):
        pass

    def copy_parameters(self):
        pass

    def save_network(self):
        pass