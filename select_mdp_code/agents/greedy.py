# greedy agent
# have to mention in paper that exhaustive trade-off calculation is computationally nasty
import random
import config
import tensorflow as tf
import environments.circle_utils as cu
import numpy as np

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
        sable_coll_list, _ = cu.check_collision(g_array, b_array)
        sable_uncoll_list = [i for i in g_list if i not in sable_coll_list]

        sable_coll_array = g_array[sable_coll_list]
        sable_uncoll_array = g_array[sable_uncoll_list]

        # print("g_list: {}".format(g_list))
        # print("b_list: {}".format(b_list))

        # print("sable_coll_list: {}".format(sable_coll_list))
        # print("sable_uncoll_list: {}".format(sable_uncoll_list))

        # print("GREEDY START")

        stop_zero = False
        stop_one = False
        n_select = 0

        while n_select < k:

            if stop_zero is True and stop_one is True:
                coin = -1
            elif stop_zero is True:
                coin = 1
            elif stop_one is True:
                coin = 0
            else:
                coin = random.randint(0, 1)

            if coin == 0:
                # print("[COIN 0] Select isolated good circles")
                # """ Narrow down sable_uncoll_list & array """

                if len(sable_uncoll_list) == 0:
                    stop_zero = True
                    # print("[STOP 0]")
                    continue

                # Find the largest isolated good circle
                action = sable_uncoll_list[np.argmax(sable_uncoll_array[:, 2])]
                largest = np.expand_dims(g_array[action], axis=0)

                # print("largest one is: {}".format(largest))
                # print("largest one has action: {}".format(action))

                # Add its index to action, increment n_select
                action_list += [action]
                n_select += 1

                # Narrow down sable_uncoll_array
                # by dropping out all the good circles that collide with the selected circle
                # including itself
                s, _ = cu.check_collision(sable_uncoll_array, largest)
                sable_uncoll_list = [i for j, i in enumerate(sable_uncoll_list) if j not in s]
                if len(sable_uncoll_list) == 0:
                    stop_zero = True
                    # print("[STOP 0]")
                    continue
                sable_uncoll_array = g_array[sable_uncoll_list]

                if action in sable_uncoll_list:
                    # Remove the chosen one from sable_coll_list & array
                    sable_uncoll_list.remove(action)
                    sable_uncoll_array = g_array[sable_uncoll_list]

                # print("=================> action_list: {}".format(action_list))

            elif coin == 1:
                # print("[COIN 1] Select good circles that collides with big bad circles")
                """ Narrow down b_big_list & array, sable_coll_list & array, sable_uncoll_list & array """

                # Get the list of bad circles whose radius is bigger than FLAGS.BIG_BAD_SIZE
                b_big_list = list(np.where(b_array[:, 2] >= FLAGS.BIG_BAD_SIZE)[0])
                b_big_array = b_array[b_big_list]

                # print("b_array: {}".format(b_array))
                # print("b_big_list: {}".format(b_big_list))

                if b_big_array.size == 0:  # No more to do in this big else statement
                    # print("[STOP 1]")
                    stop_one = True
                    continue

                while True:

                    if len(b_big_list) == 0:
                        # Make sure never we never come back
                        stop_one = True
                        break

                    # Choose the big circle to be destroyed: First in list
                    b_big_target = np.expand_dims(b_big_array[0], axis=0)
                    # print("b_big_target: {}".format(b_big_target))
                    # print("Bad target chosen")

                    # Intersecting goods
                    if len(sable_coll_list) != 0:
                        s, _ = cu.check_collision(sable_coll_array, b_big_target)
                        g_sacrifice_list = [sable_coll_list[i] for i in s]
                        g_sacrifice_array = g_array[g_sacrifice_list]
                        # print("g_sacrifice_list: {}".format(g_sacrifice_list))

                    # If there is no such goods, pop the first one out and continue loop
                    if len(sable_coll_list) == 0 or len(g_sacrifice_list) == 0:
                        b_big_list.pop(0)
                        b_big_array = b_array[b_big_list]
                        # print("new b_big_list: {}".format(b_big_list))
                        # print("No good circle to sacrifice")
                        continue

                    # If there is at least one good, choose the smallest one
                    g_target = g_sacrifice_array[np.argmin(g_sacrifice_array[:, 2]), :]
                    g_target = np.expand_dims(g_target, axis=0)
                    action = np.where(np.all(g_array == g_target, axis=1))
                    action = action[0][0]
                    action_list += [action]
                    n_select += 1
                    # print("Good target chosen")

                    # Remove colliding circles from b_bad_list & array
                    b, _ = cu.check_collision(b_big_array, g_target)
                    b_big_list = [i for j, i in enumerate(b_big_list) if j not in b]
                    b_big_array = b_array[b_big_list]

                    if len(b_big_list) == 0:
                        stop_one = True

                    # Remove colliding circles from sable_uncoll_list & array
                    if sable_uncoll_array.size != 0:
                        s, _ = cu.check_collision(sable_uncoll_array, g_target)

                        sable_uncoll_list = [i for j, i in enumerate(sable_uncoll_list) if j not in s]
                        sable_uncoll_array = g_array[sable_uncoll_list]

                    # Remove target from sable_coll_list & array
                    sable_coll_list.remove(action)
                    sable_coll_array = g_array[sable_coll_list]

                    # print("=================> action_list: {}".format(action_list))

            else:
                print("[COIN -1]")
                # Now, all selectable circles are colliding with either selected isolated or unselectable.
                # We will have to select from them anyway.

                # Need this later (urgh I hate this code) -----------------
                g_coll_list, _ = cu.check_collision(g_array, b_array)
                g_uncoll_list = [i for i in g_list if i not in g_coll_list]
                # ---------------------------------------------------------

                # Let's first choose circles that does NOT collide with selected isolated circles.
                # Such circles collide with unselectable circles => minus reward.
                # Get their list => sable_minus_list
                if sable_coll_array.size != 0:

                    isolated_action_list = [i for i in action_list if i in g_uncoll_list]

                    if len(isolated_action_list) != 0:
                        s, _, = cu.check_collision(sable_coll_array, g_array[isolated_action_list])
                    else:
                        s = []

                    sable_minus_list = [i for j, i in enumerate(sable_coll_list) if j not in s]
                    sable_minus_array = g_array[sable_minus_list]

                    # Let's drop out big circles from selectable circles - we can't afford to loose much
                    # How much can we afford to loose, then?
                    # We can't loose more than the reward that are to be obtained
                    # Therefore, I shall set is as the max size as that of previously chosen isolated circles.

                    # max size
                    if len(g_uncoll_list) != 0:
                        g_uncoll_array = g_array[g_uncoll_list]
                        max_size = np.amax(g_uncoll_array[:, 2])
                    else:
                        # no isolated circle selected so far..
                        max_size = FLAGS.MAX_RADIUS

                    # drop out big circles
                    s = list(np.where(sable_minus_array[:, 2] > max_size)[0])
                    sable_minus_list = [i for j, i in enumerate(sable_coll_list) if j not in s]
                    sable_minus_array = g_array[sable_minus_list]

                    while n_select < k:
                        # Choose among sable_minus_list until there is no more to choose

                        if len(sable_minus_list) == 0:
                            break

                        # Find the smallest of these
                        action = sable_minus_list[np.argmin(sable_minus_array[:, 2])]
                        smallest = np.expand_dims(g_array[action], axis=0)

                        # Add its index to action, increment n_select
                        action_list += [action]
                        n_select += 1

                        # Narrow down sable_minus array
                        sable_minus_list.remove(action)
                        if len(sable_minus_list) == 0:
                            break
                        sable_minus_array = g_array[sable_minus_list]

                        # print("=================> action_list: {}".format(action_list))

                while n_select < k:
                    # print("[COIN -1] RANDOM SELECT")
                    actions_available = [i for i in g_list if i not in action_list]
                    action_list += random.sample(actions_available, 1)
                    n_select += 1


        # print("")
        # print("ACTION")
        # print("action_list: {}".format(action_list))
        # print("")

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