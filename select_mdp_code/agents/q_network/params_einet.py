import random

import numpy as np
import copy as cp
import tensorflow as tf

FLAGS = tf.flags.FLAGS


class Params(object):
    def __init__(self, scope, FEATURE_SIZEs, ELEMENT_SIZEs):
        """
        scope: name of the parameters to be generated.

        FEATURE_SIZEs = 2-d array (not ndarray) with [[a0,b0,c0], [a1,b1,c1], [a2,b2,c2]] where each entry is a non-negative integer.
        There are 3 types (a,b,c) of elements for multi-type ENN. a0 denotes the number of the features in the layer 0.
        The created objects has 2 fully connected layers (the input layer :[a0,b0,c0], output layer [a2,b2,c2]).
        
        ELEMENT_SIZEs = 1-d array with [N.IN, N_CHOOSE, N.EQ]
        it just used to make scaling of the xavier initialization of EIENT

        Return: 3-dimensional list of Wg[x][y][z], Ws[x][y], b[x][y]
        x: denotes number of the layer
        y: input types of elements of the xth layer
        z: output types of elements of the xth layer    
        """
        # init the FEATURE_SIZEs and ELEMENT_SIZEs 
        self.FEATURE_SIZEs = FEATURE_SIZEs
        self.ELEMENT_SIZEs = ELEMENT_SIZEs

        # names to make the weight_name of the params 
        self.NAMES = ['in', 'eq-sel', 'eq-unsel'] 

        self.scope = scope  # To check if this agent is 'global' or not

    def get_feature_sizes(self):
        return self.FEATURE_SIZEs
        
    def generate_layers(self, k=0, trainable=True):
        """
        Generate the parameters for EINET of multiple layers.
        Wg: general parameters
        Ws: speical parameters
        b: biased Parameters

        Return:
        Wg: 3-dimensional list of Wg[x][y][z]
        Ws: 2-dimensional list of Ws[x][y]
        b: 2-dimensional list of b[x][y]

        x: denotes number of the layer
        y: input types of elements of the xth layer
        z: output types of elements of the xth layer  
        """
        [Wg, Ws, b] = [[], [], []]  # list for layers
        layer_name = ''  # name for weights per layers

        for x in range(len(self.FEATURE_SIZEs)-1):
            # generate layer name
            layer_name = self.scope
            if trainable:
                layer_name += '_main'
            else:
                layer_name += '_target'
            layer_name += '_tran_' + str(k) + '_layer_' + str(x)

            # append Wg, Ws, b 
            Wg_temp, Ws_temp, b_temp = self._generate_layer(
                layer_name,self.FEATURE_SIZEs[x],self.FEATURE_SIZEs[x+1],self.ELEMENT_SIZEs,trainable)
            Wg.append(Wg_temp)
            Ws.append(Ws_temp)
            b.append(b_temp)
        
        return Wg, Ws, b

    def _generate_layer(self, layer_name, in_features, out_features, element_sizes, trainable):
        """
        Generate the parameters for EINET of 1-layers.
        
        Return: 2-dimensional list of xth layer param[y][z]
        y: input types of elements of the xth layer
        z: output types of elements of the xth layer    
        """
        Wg = []
        Ws = []
        b = []

        # temporal values for names 
        temp_name = ''
        temp_Wg = [] 

        # generate Wg general weights
        for i in range(len(in_features)):
            for j in range(len(out_features)):
                temp_name = layer_name + '_Wg_' + self.NAMES[i] + '_' + self.NAMES[j]
                temp_Wg.append(self._generate_param(temp_name, in_features[i], out_features[j], element_sizes[j], trainable))
            Wg.append(temp_Wg)
            temp_Wg = []

        # generate Ws
        for i in range(len(in_features)):
            temp_name = layer_name + '_Ws_' + self.NAMES[i]
            Ws.append(self._generate_param(temp_name, in_features[i], out_features[i], element_sizes[i], trainable))

        # generate b        
        for i in range(len(in_features)):
            temp_name = layer_name + '_b_' + self.NAMES[i]
            b.append(self._generate_param(temp_name, 1,  out_features[i], element_sizes[i], trainable))

        return [Wg, Ws, b]
    
    def _generate_param(self, param_name, in_feature, out_feature, out_element, trainable):
        """
        Generate param as matrix [in_feature, out_feature] with param_name, out_element is the number of elements used for Xavier_Initialization

        Return: a weight matrix with dim [in_feature, out_feature]
        """
        # xavier_initial_bound
        if not out_element == 0:
            bound_uniform_xavier = np.power(1/np.float(out_element), FLAGS.XAVIER) * np.sqrt(6/np.float(in_feature+out_feature))
        else:
            bound_uniform_xavier = np.sqrt(6 / np.float(in_feature + out_feature))

        # get param with matrix form
        param = tf.get_variable(param_name, [in_feature, out_feature], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-bound_uniform_xavier, maxval=bound_uniform_xavier), trainable=trainable)

        return param

# just for test copy (do not need to use it in params_einet.py)
def get_copy_ops(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


if __name__ == '__main__':
    sess = tf.Session()
    Parma_einet = Params('global', [[1,4000,3], [2,2,2], [1,5,3]], [40000,40000,1])
    K = 3
    Wg_main = [None] * K
    Ws_main = [None] * K
    b_main = [None] * K

    for i in range(K):
        Wg_main[i], Ws_main[i], b_main[i] = Parma_einet.generate_layers(i, True)

    Wg_target = [None] * K
    Ws_target = [None] * K
    b_target = [None] * K
    
    for i in range(K):
        Wg_target[i], Ws_target[i], b_target[i] = Parma_einet.generate_layers(i, False)
    print('length', len(Wg_target))
    sess.run([tf.global_variables_initializer()])

    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'))
    # print('main', sess.run(tf.global_variables('global_main')))
    # print('target', sess.run(tf.global_variables('global_target')))


    # # copy test

    copy = get_copy_ops('global_main', 'global_target')
    
    # sess.run(copy)

    # print('main', sess.run(tf.global_variables('global_main')))
    # print('target', sess.run(tf.global_variables('global_target')))
