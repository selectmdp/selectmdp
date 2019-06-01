import tensorflow as tf
import numpy as np
import copy as cp

import os
import sys

import environments.circle_utils as cu
# import circle_utils as cu # if you want directly test max_area

FLAGS = tf.flags.FLAGS
np.random.seed(FLAGS.SEED)

class Environment(object):
    def __init__(self, N_EQ = FLAGS.N_EQUIVARIANT, N_IN = FLAGS.N_INVARIANT, train_mode = True):
        """
        Initialize environment system
        """
        self.N_EQ = N_EQ
        self.N_IN = N_IN # get the number of elements
        self.reset()
        # cu = Circle_Util()

    def reset(self):
        """Reset the states
        """
        self.generate_state()
        self.generate_update_matrix()

        return None

    def step(self, action):
        """Feed the action, and make transtitions

        Return reward
        """
        # updates the equi, invariant circles before actions
        self.update_before_collide(action) 
        # update the circles with selected action, get reward
        self.eq_collision_list, self.in_collision_list =  self.update_with_action(action[0])   
        # generate new circle circles
        self.update_select_eq_elements(action[0])
        if self.N_IN>0 and self.in_collision_list != []:
            # deleted the invariant circles
            self.update_inv_collision(self.in_collision_list)
        self.update_clip() # take care the bound of the circles
        self.update_state() # just update the state

        return self.reward/np.pi, self.rad_selected
        # return self.reward/np.pi


    def get_collision(self):
        """Get the collided circles

        Return: 
        eq_coll=collision among selected equivariant circles
        in_coll=collision among selected invariant circles
        """
        return self.eq_collision_list, self.in_collision_list

    def generate_state(self):
        """
        Make the empty state matries of elements, -both equivariant and invariant.  
        """
        # initialize the number of elements, features, if
        self.FEAT_EQ = FLAGS.N_FEATURES # FEAT_EQ = 3, (x,y,r), r denotes radius
        self.FEAT_IN = FLAGS.N_FEATURES # FEAT_IN = 3, (x,y,r), r denotes radius

        # generate state matrix for equivariant elements
        self.eq_matrix = np.empty([self.N_EQ, self.FEAT_EQ])
        self.eq_matrix = cu.generate_xy_elements(self.N_EQ)
        self.eq_matrix[:,2] = cu.generate_r_elements(self.N_EQ)

        # generating invariant matrix if self.N_IN>0         
        # if self.N_IN > 0:
        self.in_matrix = np.empty([self.N_IN, self.FEAT_IN])
        self.in_matrix = cu.generate_xy_elements(self.N_IN)
        self.in_matrix[:,2] = cu.generate_r_elements(self.N_IN) 

        # generate state as dictionary
        self.state = {}
        self.state['equivariant_array'] = self.eq_matrix
        # if self.N_IN > 0:
        self.state['invariant_array'] = self.in_matrix

        #TODO: considered the action is also added. 
       
        return None

    def generate_update_matrix(self):
        """
        Generate self.eq_update, self.in_update which used in transition of the unselected element.
        """
        # noise matries used for unselected transitions ex) eq_mat += eq_update
        self.MAX_NOISE = FLAGS.MAX_NOISE # noise scale
        self.eq_update = np.empty([self.N_EQ, self.FEAT_EQ])

        # usage) self.eq_update -= self.EQ_HALF_UPDATE
        self.EQ_HALF_UPDATE = 0.5 * np.ones([self.N_EQ, self.FEAT_EQ])
        
        #  just constant matrix for making the epectected value of  eq_update, in_update to be zero 
        if self.N_IN>0:
            self.in_update = np.empty([self.N_IN, self.FEAT_IN])
            self.IN_HALF_UPDATE = 0.5 * np.ones([self.N_IN, self.FEAT_IN])

    def get_state(self):
        """Return the current state
        """
        self.update_state()
        return self.state

    def get_reward(self):
        
        return self.reward

    def plot_selected_elements(self, action):
        """Draw the selected circles
        """
        pass

    def update_with_action(self,action):
        """update with action

        Return:
        eq_collision_list: collisioned equivariant circles
        in_collision_list: collisioned invariant circles
        """
        self.reward = 0 # reward for this step

        # generate lists for collision between equi-circle and inv-circles
        action = np.array(action)
        in_collision_list = [] # collisioned invariant circles
        eq_collision_list = [] # collisioned equivariant circles

        # estimate negative reward (coll_negative_reward) by the area of collisioned good circles
        if self.N_IN>0:
            coll_negative_reward, eq_collision_list,  in_collision_list  = self.eq_in_collision(action)
            self.reward = self.reward + coll_negative_reward

        # get the positive reward among equi-circles without collision
        positive_reward_eq_remained = self.eq_remain_collision(action, eq_collision_list)
        self.reward = self.reward + positive_reward_eq_remained

        return eq_collision_list, in_collision_list

    def update_before_collide(self, action):
        """update the informations before selection
        """
        # general updates for all equi+invariant circles 
        self.update_noise()

        # expand the equi circles radius
        self.update_eq_expand()

        if self.N_IN>0:
            # expand the inv circles radius
            self.update_in_expand()

        # move with the actions
        self.eq_matrix[action[0,action[1]==1],0]-= FLAGS.ACTION_MOVE_DIST #left
        self.eq_matrix[action[0,action[1]==2],1]+= FLAGS.ACTION_MOVE_DIST #up
        self.eq_matrix[action[0,action[1]==3],0]+= FLAGS.ACTION_MOVE_DIST #right
        self.eq_matrix[action[0,action[1]==4],1]-= FLAGS.ACTION_MOVE_DIST #down

        return None

    def update_select_eq_elements(self, action):
        """Update the selected elements
        """
        # selected (x,y,r) initialization
        # print('self.eq_matrix', self.eq_matrix)
        # print('action', action)
        # print('self.eq_matrix[action,[0,1]]', self.eq_matrix[:,[0,1]])
        # print(' np.ones([len(action), 2])) ',  np.ones([len(action), 2]))
        self.eq_matrix[action,:] = cu.generate_xy_elements(len(action))
        self.eq_matrix[action,2] = cu.generate_r_elements(len(action),small_start=True)

        return None

    def update_noise(self):
        """Update unselected both equivariant and invariant elements 
        """
        self.eq_update =FLAGS.MAX_NOISE*(np.random.rand(self.N_EQ, self.FEAT_EQ) - self.EQ_HALF_UPDATE)
        # self.eq_update =FLAGS.MAX_NOISE*(self.EQ_HALF_UPDATE )

        self.eq_matrix = self.eq_matrix + self.eq_update
       
        if self.N_IN > 0:
            self.in_update = FLAGS.MAX_NOISE*(np.random.rand(self.N_IN, self.FEAT_IN) - self.IN_HALF_UPDATE)
            self.in_matrix = self.in_matrix + self.in_update
        
        return None

    def update_clip(self):
        """Consider the bound of x-axis, y-axis, radius
        """
        # clip (x,y,r)
        self.eq_matrix[:,[0,1]] = np.clip(self.eq_matrix[:,[0,1]], -FLAGS.UB_CENTER, FLAGS.UB_CENTER)
        self.eq_matrix[:,2] = np.clip(self.eq_matrix[:,2], 0, FLAGS.MAX_RADIUS)

        if self.N_IN > 0:
            self.in_matrix[:,[0,1]] = np.clip(self.in_matrix[:,[0,1]], -FLAGS.UB_CENTER, FLAGS.UB_CENTER)
            self.in_matrix[:,2] = np.clip(self.in_matrix[:,2], 0, FLAGS.MAX_RADIUS)

        return None

    def update_state(self):
        """Update the state to current matrix. HS: Dictionary can not be automatically updated so we need this right?
        """
        self.state['equivariant_array'] = self.eq_matrix
        if self.N_IN > 0:
            self.state['invariant_array'] = self.in_matrix
        return None

    def eq_in_collision(self,action):
        """Find the list of the good circles which collapse
        the collided equivariant circle 
        return the list of the good circles, bad circles which are collided and 
        
        Return: the negative reward = - (area of collided good circle) 
        """
        # parse the selected equviraint elements
        # try:
        eq_selected = self.eq_matrix[action,:]
        self.rad_selected = eq_selected[:,2]
        # print('self.rad_selected',self.rad_selected)
        # exit(0)
        # except:
        #     print('action',action)
        #     print('self.eq_matrix',self.eq_matrix)

        # divide the selected eq, and in with x,y,r
        eq_coll, in_coll = cu.check_collision(eq_selected, self.in_matrix)

        # compute the negative reward for equivariant circles
        coll_negative_reward =  - np.pi * np.sum(np.multiply(eq_selected[eq_coll,2], eq_selected[eq_coll,2])) 
        coll_negative_reward = - cu.sum_circle_area(eq_selected[:,2], eq_coll)
       
        return coll_negative_reward, action[eq_coll], in_coll

    def eq_remain_collision(self, action, eq_collision_list):
        """Check the collision among the good remained circles remained and get the positive reward
        """
        eq_remain_matrix = self.eq_matrix[np.setdiff1d(action,eq_collision_list),:]
        positive_remain_reward = 0
        if len(eq_remain_matrix): # if eq_remain is not empty
            eq_remain_coll, _ = cu.check_collision(eq_remain_matrix, eq_remain_matrix)
            eq_remain = np.setdiff1d(np.arange(len(eq_remain_matrix)), eq_remain_coll)
            
            # find uncollied equivariant elements
            positive_remain_reward = cu.sum_circle_area(eq_remain_matrix[:,2], eq_remain)

        return positive_remain_reward

    def update_inv_collision(self, in_collision_list):
        """Re initialize the collisioned invariant circles
        """
        self.in_matrix[in_collision_list,:] = cu.generate_xy_elements(len(in_collision_list)) 
        self.in_matrix[in_collision_list,2] = cu.generate_r_elements(len(in_collision_list), small_start=True)

        return None

    def update_eq_expand(self):
        """Expand the equivariant circle's radius r
        """
        EXPAND_eq = 0.8 * FLAGS.MAX_RADIUS * FLAGS.N_CHOOSE/np.float(self.N_EQ)
        # EXPAND_eq=0 # NOTE HS: just guess for expension, I want slow expension rate so the heurstic would destroy all middle size circle and going inefficient with training step 1000 
        self.eq_matrix[:,2] = self.eq_matrix[:,2] + self.update_expand(self.N_EQ, EXPAND_eq)

        # resize the radius if maximize more
        self.eq_matrix[:,2][self.eq_matrix[:,2]>FLAGS.MAX_RADIUS] = FLAGS.INIT_RADIUS

        return None

    def update_in_expand(self):
        """Expand the invariant circles' radius r
        further more erase the circle after the large size
        """
        self.in_matrix[:,2] = self.in_matrix[:,2] + self.update_expand(self.N_IN,EXPAND_RATE=FLAGS.EXPAND_CONST)
        
        return None

    def update_expand(self, NUMBER_ELEMENTS, EXPAND_RATE):
        """Expand the equivariant circle's radius r
        """
        expand_r = EXPAND_RATE * (0.45 + 0.1 * np.random.rand(NUMBER_ELEMENTS))
        
        # test
        # expand_r = EXPAND_RATE * np.ones(NUMBER_ELEMENTS)
        return expand_r
    
    def _sort_state(self):
        """Sort the state with largest radius
        """
        self.eq_matrix = sorted(self.eq_matrix,
            key=lambda x: x[-1], reverse=True)
        self.eq_matrix = np.array(self.eq_matrix)
        self.in_matrix = sorted(self.in_matrix,
            key=lambda x: x[-1], reverse=True)
        self.in_matrix = np.array(self.in_matrix)


        return None
    
    def _shuffle_state(self):
        """Shuffle the states with pure random order
        """
        np.random.shuffle(self.eq_matrix)
        np.random.shuffle(self.in_matrix)

        return None



if __name__ == '__main__':
    Env = Environment()
    print('before select')
    # print('Env.eq_matrix', Env.eq_matrix)
    # print('Env.in_matrix', Env.in_matrix)
    # print('update whole', Env.update_noise())
    # # print('Env.eq_matrix', Env.eq_matrix)
    # # print('Env.in_matrix', Env.in_matrix)
    # # print('update select',Env.update_select_eq_elements([1]))
    # # print('Env.eq_matrix', Env.eq_matrix)
    # # print('Env.in_matrix', Env.in_matrix)
    # # print('update expand',Env.update_in_expand())
    # # print('Env.eq_matrix', Env.eq_matrix)
    # # print('Env.in_matrix', Env.in_matrix)
    # print('after clip',Env.update_clip())
    # # print('Env.eq_matrix', Env.eq_matrix)
    # # print('Env.in_matrix', Env.in_matrix)
    # print('after_select', Env.get_state())
    # print('Env.get_reward()', Env.get_reward(np.array([0,1,2,3,4])))
    # print('Env.eq_remain_collision', Env.eq_remain_collision)
    print('get_state', Env.get_state())
    print('with action', Env.step(np.array([0,1,2,3,4])))