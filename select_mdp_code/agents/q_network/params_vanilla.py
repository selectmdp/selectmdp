import random

import numpy as np
import copy as cp
import tensorflow as tf
import config

FLAGS = tf.flags.FLAGS


class Params(object):
    def __init__(self, scope, FEATURE_SIZEs, ELEMENT_SIZEs):
        """
        scope: name of the parameters to be generated.

        FEATURE_SIZEs = 2-d array (not ndarray) with [[a0,b0,c0], [a1,b1,c1], [a2,b2,c2]] where each entry is a non-negative integer.
        There are 3 types (a,b,c) of elements for multi-type ENN. a0 denotes the number of the features in the layer 0.
        The created objects has 2 fully connected layers (the input layer :[a0,b0,c0], output layer [a2,b2,c2]).

        ELEMENT_SIZEs = 1-d array with [N.IN, N_CHOOSE, N.EQ]
        """

        # init the FEATURE_SIZEs and ELEMENT_SIZEs
        self.FEATURE_SIZEs = FEATURE_SIZEs
        self.ELEMENT_SIZEs = ELEMENT_SIZEs
        # self.init_element_sizes = [FLAGS.N_INVARIANT, 0, FLAGS.N_EQUIVARIANT]
        self.N_LAYERS = len(self.FEATURE_SIZEs)

        # names to make the weight_name of the params
        # self.NAMES = ['in', 'eq-sel', 'eq-unsel']

        self.scope = scope  # To check if this agent is 'global' or not

    def generate_layers(self, k=0, trainable=True):
        """
        Generate the parameters for Q network (vanilla) of multiple layers.

        Return:
            W: list of arrays
            b: list of arrays
        """

        W = []
        b = []

        element_sizes = (FLAGS.N_INVARIANT, k, FLAGS.N_EQUIVARIANT - k)

        for x in range(self.N_LAYERS - 1):
            # generate layer name
            layer_name = self.scope
            if trainable:
                layer_name += '_main'
            else:
                layer_name += '_target'
            layer_name += '_tran_' + str(k) + '_layer_' + str(x)

            # append W and b
            W_temp, b_temp = self._generate_layer(layer_name,
                                                  self.FEATURE_SIZEs[x],
                                                  self.FEATURE_SIZEs[x+1],
                                                  element_sizes,
                                                  trainable,
                                                  last=(x == (self.N_LAYERS - 2)))

            W.append(W_temp)
            b.append(b_temp)

        return [W, b]

    def _generate_layer(self, layer_name, in_features, out_features, element_sizes, trainable, last=False):
        """
        Generate the parameters of 1-layer network.

        in_features = [a0, a1, a2]
        out_features = [b0, b1, b2]
        element sizes = [e1, e2, e3]

        Return:
            W: array (not ndarray) size (in_dim, out_dim)
            b: array (not ndarray) size (1, out_dim)
        """

        in_dim = np.dot(in_features, element_sizes)

        if last:
            out_dim = out_features[-1] * element_sizes[-1]  # only equi-select
        else:
            out_dim = np.dot(out_features, element_sizes)

        # # xavier_initial_bound
        # if not out_dim == 0:
        #     bound_uniform_xavier = np.power(1 / np.float(out_dim), 1) * np.sqrt(6 / np.float(in_dim + out_dim))
        # else:
        #     bound_uniform_xavier = np.sqrt(6 / np.float(in_dim + out_dim))

        # TODO: trainable=trainable
        W = tf.get_variable(layer_name+'_W',
                            [in_dim, out_dim],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(layer_name+'_b',
                            [1, out_dim],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        return [W, b]
