import random

import numpy as np
import copy as cp
import tensorflow as tf

# FLAGS = tf.flags.FLAGS

class Replay_Buffer(object):
    """
    class for the replay buffer
    """
    def __init__(self, BUFF_SIZE=5, BATCH_SIZE =2, N_EQ=5, N_IN=3, N_CHOOSE=2, N_ACT = 1, IN_FEAT = 3, EQ_FEAT = 3):
        """
        initialize several replay buffers from 0 to (N_CHOOSE-1)th replay buffer, some pointers, and several parameters
        for each ith buffer includes 9 different subbuffers ["in", "equi_selected", "equi_unselected", "action_elements", "act_subact", "reward", "next_in", "next_eq_selected", "next_eq_unselected"]
        where action_elements: chosen elements, act_subact: subaction of the elements
        """
        self._generate_values(BUFF_SIZE, BATCH_SIZE ,N_EQ, N_IN, N_CHOOSE, N_ACT = N_ACT, IN_FEAT = IN_FEAT, EQ_FEAT = EQ_FEAT)
        self._generate_replay_buffers()
        return None

    def _generate_values(self, BUFF_SIZE=5, BATCH_SIZE =2, N_EQ=5, N_IN=3, N_CHOOSE=2, N_ACT = 1, IN_FEAT = 3, EQ_FEAT = 3):
        """
        generate some initial values related to the job
        """ 
        # set the initial values
        self.BUFF_SIZE = int(BUFF_SIZE)
        self.BATCH_SIZE = int(BATCH_SIZE)
        self.N_EQ = int(N_EQ)
        self.N_IN = int(N_IN)
        self.N_CHOOSE = int(N_CHOOSE)
        self.N_SUBACT = int(N_ACT) # number of subactions, (default 1: in circle selection)
        self.IN_FEAT = int(IN_FEAT)
        self.EQ_FEAT = int(EQ_FEAT)


        self.check_correct_input()
        return None
    
    def check_correct_input(self):
        """
        Check whether the initial inputs are correct
        """
        if self.BATCH_SIZE>self.BUFF_SIZE:
            print('buff size should be larger than batch size')
            exit(-1)
        
        if self.N_CHOOSE>self.N_EQ:
            print('equivariant elements should be larger than number of actions')
            exit(-1)

        return None

    def _generate_replay_buffers(self):
        """
        generate the self.N_CHOOSE replay buffers and pointers
        self.total_bufs[i]:= replay buffer for ith actions 
        """
        ## buffers contains the whole 
         #TODO: need to change self.N_CHOOSE

        ## pointer to imply self.pointer_buf th elements in self.total_bufs[self.pointer_buf]
        self.pointer_buf = 0


        ## initialize the index of the different buffers as constant
        self.BUF_STRING = ["in", "equi_selected", "equi_unselected", "act_select", "act_subact","reward", "next_in", "next_eq_selected", "next_eq_unselected"] 
        # TODO: erase equi_selected, equi_unselected same 
        self.BUF_TYPE = len(self.BUF_STRING) # number of the buffer type 
        [self.IN_BUF, self.EQ_SEL_BUF, self.EQ_UNSEL_BUF, self.ACT_SEL_BUF, self.ACT_SUB_BUF, self.REWARD_BUF, self.NEXT_IN_BUF, self.NEXT_EQ_SEL_BUF, self.NEXT_EQ_UNSEL_BUF] = range(self.BUF_TYPE)
        ## just string for easy debug

        self.TRANS_TYPE = 6 # number of trans type (eq_select, eq_unselect --> eq )
        [self.IN_TRANS, self.EQ_TRANS, self.ACT_TRANS, self.REWARD_TRANS, self.NEXT_IN_TRANS, self.NEXT_EQ_TRANS] = range(self.TRANS_TYPE)

        
        self.total_bufs = self._generate_ith_buffer()

    def _generate_ith_buffer(self): # TODO only one buffer needs
        """
        generate replay buffer for ith transition.
        --------
        return the ith buffer (in, eq_select, eq_unselect, act_select, act_subact, reward, next_in, next_eq_select, next_eq_unselect)
        """
        buf = [None] * self.BUF_TYPE

        # invariant buffer
        buf[self.IN_BUF] = -1.5 * np.ones([self.BUFF_SIZE, self.N_IN, self.IN_FEAT])
        buf[self.NEXT_IN_BUF] = -1.5 * np.ones([self.BUFF_SIZE, self.N_IN, self.IN_FEAT])

        # equvariant and selected elements (feature includes chosen actions)
        buf[self.EQ_SEL_BUF] = -1.5 * np.ones([self.BUFF_SIZE,0, self.EQ_FEAT+self.N_SUBACT])

        # equivariant unselect buff, only equivariant features
        buf[self.EQ_UNSEL_BUF] = -1.5 * np.ones([self.BUFF_SIZE,self.N_EQ,self.EQ_FEAT])

        # action buffer: action is one-hot vecotr among (N_EQ-ith selectable elements)
        buf[self.ACT_SEL_BUF] = [None] *self.N_CHOOSE
        buf[self.ACT_SUB_BUF]  = [None] * self.N_CHOOSE
        for ith in range(self.N_CHOOSE):
            buf[self.ACT_SEL_BUF][ith] = -1.5 * np.ones([self.BUFF_SIZE, self.N_EQ])
            buf[self.ACT_SUB_BUF][ith] = -1.5 * np.ones([self.BUFF_SIZE, self.N_SUBACT])

        # reward buffer
        buf[self.REWARD_BUF] = -1.5 * np.ones([self.BUFF_SIZE,1])

        # when final transition (next equi-buffer has different shape)
        buf[self.NEXT_EQ_SEL_BUF] =  -1.5 * np.ones([self.BUFF_SIZE, 0, self.EQ_FEAT+self.N_SUBACT]) 
        buf[self.NEXT_EQ_UNSEL_BUF] = -1.5 * np.ones([self.BUFF_SIZE, self.N_EQ ,self.EQ_FEAT])

        return buf

    def add_trans(self, trans): 
        """
        add transition to the replay buffer, it slices the transitions properly to N_CHOOSE different number of buffers
        ------------
        trans: input (s,a,r,s') 
        6 numpy array (in, eq, act (with order +sub act with [2, N_CHOOSE] array), reward, next_in, next_eq)
        trans convert to a list that composed of the 8 values 9 numpy-arrays or float (in, eq_select, eq_unselect, act_select, act_subact, reward, next_in, next_eq_select, next_eq_unselect)
        """
        # check whether correctely works
        for i in range(len(trans)):
            trans[i] = np.array(trans[i])
        
        # add sub-action (with one-hot) for each selected equi-elements 
        # print('trans[self.ACT_TRANS]', trans[self.ACT_TRANS])
        # Action integer convert
        trans[self.ACT_TRANS] = np.array(trans[self.ACT_TRANS], np.int32)
        # change fomr with two lines
        trans[self.ACT_TRANS] = cp.deepcopy(np.reshape(trans[self.ACT_TRANS], (2,-1)))
            # print(trans[self.ACT_TRANS], 'trans[self.ACT_TRANS]')
        
        # save the trans in the proper form for each ith_buffer
        # TODO:erase until interfere
        # for ith in range(self.N_CHOOSE):
        sliced_trans = self.slicing(trans)
        self.interfere_trans(sliced_trans)

        # grow and reset if needed
        self.pointer_buf +=1 
        if self.pointer_buf == self.BUFF_SIZE:
            self.pointer_buf = 0

            # for i in range(self.N_CHOOSE):
            #     for j in range(len(self.total_bufs[i])):
            #         print(str(i) + 'th action buff type '+str(self.BUF_STRING[j]), self.total_bufs[i][j])
            # exit(0)

    def _generate_act_one_hot(self, trans): #TODO erase
        """Reorder the equivariant elements, and actions proper form. [eq-sel, eq-unsel, acts] (adding one-hot to eq-select, changes the act_sel (if the first eq-element disappears, the remaining eq-select is different) among unselect eq-elements)
        """
        trans_temp = [None] * 3 # [eq-sel, eq-unsel, acts]
        [EQ_SEL, EQ_UNSEL, ACT] = range(3) # create integers
        trans_temp[EQ_SEL] = np.zeros([0, self.EQ_FEAT+self.N_SUBACT]) ## initial is with zero dim
        trans_temp[EQ_UNSEL] = cp.deepcopy(trans[self.EQ_TRANS])
        trans_temp[ACT] = cp.deepcopy(trans[self.ACT_TRANS])

        return trans_temp

    def interfere_trans(self, sliced_trans):
        """
        interfer the sliced_trans for ith action or ith transitions into the bufs related to ith (0<=i<N_CHOOSE)
        buf_type:= (in_buf, eq_select_buf, eq_unselect_buf, a_ith, reward, next_in_bf, next_eq_select, next_eq_unselect)
        """ 
        for buf_type in range(len(self.total_bufs)):
            # print('ith',ith)
            # print('buf_type',buf_type)
            # print('self.total_bufs[ith][buf_type][self.pointer_buf]',np.shape(self.total_bufs[ith][buf_type][self.pointer_buf]))

            # print('sliced_trans[buf_type]',np.shape(sliced_trans[buf_type]))
            if buf_type == self.ACT_SEL_BUF or buf_type == self.ACT_SUB_BUF:
                for ith in range(self.N_CHOOSE):
                    # print('sliced_trans[buf_type]', sliced_trans[buf_type][ith])
                    self.total_bufs[buf_type][ith][self.pointer_buf] = sliced_trans[buf_type][ith]
            else:
                self.total_bufs[buf_type][self.pointer_buf] = sliced_trans[buf_type]
               
        return None

    def _one_hot(self, whole_dim, chosen):
        """Return the one-hot vector with whole_dim + chosen elements
        """
        one_hot = np.zeros([1, whole_dim], dtype = np.int32)
        # print('whole_dim',whole_dim)
        # print('chosen', chosen)
        one_hot[0,chosen] = int(1)
        # print('one_hot',one_hot)

        return one_hot
    
    def slicing(self, trans): #TODO:erase
        """
        slices the equivariant matrix properly
        ith_buffer related to ith replay buffer it discriminate the size of the buffer. 
        Input:
        6 numpy array (in, eq, act, reward, next_in, next_eq)
        trans convert to a list that composed of the 8 values 8 numpy-arrays or float (in, eq_select, eq_unselect, act, reward, next_in, next_eq_select, next_eq_unselect)
        -----------
        Return: the sliced transition for proper 
        """
        # normal transitions (when ith< N_CHOOSE-1)
        sliced_trans_i = [None] * self.BUF_TYPE

        # invariant sliced (current and future)
        sliced_trans_i[self.IN_BUF] = trans[self.IN_TRANS]
        # current equivariant - select, unselect
        sliced_trans_i[self.EQ_SEL_BUF] = np.zeros([0, self.EQ_FEAT +self.N_SUBACT])
        sliced_trans_i[self.EQ_UNSEL_BUF] = trans[self.EQ_TRANS]

        
        # print('trans_temp[EQ_USEL]', trans_temp[EQ_UNSEL])
        # print('trans_temp[ACT][0,0]', trans_temp[ACT][0,0])
        # one-hot vectors for eq_sel, eq_subact
        # print('trans_tem[ACT]', trans_temp[ACT])
        # print('trans_temp[ACT][0,0]', trans_temp[ACT][0,0])
        # print('trans_temp[ACT][1,0]', trans_temp[ACT][1,0])
        sliced_trans_i[self.ACT_SEL_BUF] = [None] *self.N_CHOOSE
        sliced_trans_i[self.ACT_SUB_BUF] = [None] *self.N_CHOOSE

        for ith in range(self.N_CHOOSE):
            act_sel = self._one_hot(self.N_EQ, trans[self.ACT_TRANS][0,ith])
            # print('self.N_SUBACT', self.N_SUBACT) 
            act_sub = self._one_hot(self.N_SUBACT, trans[self.ACT_TRANS][1,ith])
        
        # set actions with one-hot vector slice
            sliced_trans_i[self.ACT_SEL_BUF][ith] = act_sel
            sliced_trans_i[self.ACT_SUB_BUF][ith] = act_sub

        # print('act_sel, act_sub',act_sel, act_sub)

        # print('whole, target,act_trans', whole_size, target, act_trans)

        # generate feature + subaction vector for currentely chosen elements
        # print('eq_sel_feat',eq_sel_feat)
        # exit(0)
        

        # eq_sel_added_act = np.reshape(eq_sel_added_act, [1,-1])
        # print('eq_sel_added_act', eq_sel_added_act)

        # update trans_temp
        # print(trans_temp[EQ_SEL])
        # print(trans_temp[EQ_SEL].shape())

        # print(eq_sel_added_act)
        # reward # usually the zero reward
        sliced_trans_i[self.REWARD_BUF] = 0

        # final transitions, transitions are different
        # if ith == self.N_CHOOSE-1:
        sliced_trans_i[self.REWARD_BUF] = trans[self.REWARD_TRANS]
        sliced_trans_i[self.NEXT_IN_BUF] = trans[self.NEXT_IN_TRANS]
        sliced_trans_i[self.NEXT_EQ_SEL_BUF] = np.ones([0, self.EQ_FEAT + self.N_SUBACT]) # just for empty
        sliced_trans_i[self.NEXT_EQ_UNSEL_BUF] = trans[self.NEXT_EQ_TRANS]

        return sliced_trans_i

    # def shuffle_buffer(self, ith): #TODO: do not requires
        # """
        # shuffles states in the replay buffer. shuffle the order samples in the ith buffer 
        # """
        # ith = int(ith)
        # np.random.shuffle(self.shuffled_order)
        # for buf_type in range(len(self.total_bufs[ith])):
        #     self.total_bufs[ith][buf_type] = self.total_bufs[ith][buf_type][self.shuffled_order]
        # return None
    
    def save_buffer(self, path): 
        """Save the replay buffers to npy
        """
        #TODO:change the N_CHOOSE
        # for ith in range(self.N_CHOOSE):
        for buf_type in range(self.BUF_TYPE):
            if not buf_type == self.ACT_SEL_BUF or buf_type== self.ACT_SUB_BUF:
                np.save(path + 'buf_'+str(buf_type), self.total_bufs[buf_type])
            else:
                for ith in range(self.N_CHOOSE):
                    np.save(path + 'buf_'+str(buf_type), self.total_bufs[buf_type][ith])
                
        
    def load_buffer(self, path, buffer_point=0):
        """Load the replay buffers for npy
        """
        #TODO:change the N_CHOOSE
        for buf_type in range(self.BUF_TYPE):
            if not buf_type == self.ACT_SEL_BUF or buf_type== self.ACT_SUB_BUF:
                np.load(path + 'buf_'+str(buf_type)+'.npy')
            else:
                for ith in range(self.N_CHOOSE):
                    np.load(path + 'buf_'+str(buf_type)+'.npy')

        # for ith in range(self.N_CHOOSE):
        #     for buf_type in range(self.BUF_TYPE):
        #         self.total_bufs[ith][buf_type] = np.load(path + 'buf_'+str(ith)+'_'+str(buf_type)+'.npy')

        self.pointer_buf = buffer_point % self.BUFF_SIZE
        

    def get_batch(self):
        """
        randomly sample out the batch from the replay buffer.
        ------
        Return:list of randomly chosen batches 
        batch = [batch[0],..., batch[N_CHOOSE-1]]
        batch[ith] = 
        [[in_mat[ith][batch_index], eq_select_mat[ith][batch_index], eq_unselect_mat[ith][batch_index], action[ith][batch_index], reward[ith][batch_index], in_next_mat[ith][batch_index], eq_select_next_mat[ith][batch_index], eq_unselect_next_mat[ith][batch_index]] 
        """
        batch_index = np.random.choice(self.BUFF_SIZE, self.BATCH_SIZE, False)
        
        #TODO change total_buf[ith]
        batch = []
        for buf_type in range(self.BUF_TYPE):
            if buf_type == self.ACT_SEL_BUF or buf_type==self.ACT_SUB_BUF:
                act_batch = []
                for ith in range(self.N_CHOOSE):
                    act_batch.append(self.total_bufs[buf_type][ith][batch_index])
                batch.append(act_batch)
            else:
                batch.append(self.total_bufs[buf_type][batch_index])
        # batch = [[buf_type[batch_index] for buf_type in self.total_bufs[ith]] for ith in range(self.N_CHOOSE)]

        return batch

if __name__ == "__main__":
    ## initialize the test hyperparameters
    BUFF_SIZE = 5
    BATCH_SIZE  = 3
    N_EQ  = 5
    N_IN  = 2
    N_CHOOSE  = 3
    N_ACT   = 1
    IN_FEAT  = 3
    EQ_FEAT  = 3

    trans = [np.ones([N_IN,IN_FEAT]), 2 * np.ones([N_EQ, EQ_FEAT]), np.arange(N_CHOOSE), 3, 4*np.ones([ N_IN, IN_FEAT]), 5 * np.ones([N_EQ, EQ_FEAT])]

    ## initialize the buffer
    Buffer = Replay_Buffer(BUFF_SIZE=BUFF_SIZE, BATCH_SIZE=3, N_EQ=N_EQ, N_IN=N_IN, N_CHOOSE=N_CHOOSE, N_ACT= N_ACT, IN_FEAT= IN_FEAT, EQ_FEAT=EQ_FEAT)

    ## test the initial buffer
    for i in range(N_CHOOSE):
        for j in range(len(Buffer.total_bufs[i])):
            print(str(i) + 'th action buff type '+str(Buffer.BUF_STRING[j]), Buffer.total_bufs[i][j])

    ## fill up the buffer
    print('1sttrans[ACT]', trans[2])
    Buffer.add_trans(cp.deepcopy(trans))
    print('2ndtrans[ACT]', trans[2])
    Buffer.add_trans(cp.deepcopy(trans))
    Buffer.add_trans(cp.deepcopy(trans))
    Buffer.add_trans(cp.deepcopy(trans))


    ## fill up the buffer differently
    trans1 = [6 * np.ones([N_IN,IN_FEAT]), 7*np.ones([N_EQ, EQ_FEAT]), np.array([0,3,2]), 8, 9*np.ones([N_IN, IN_FEAT]), 10 * np.ones([N_EQ, EQ_FEAT])]

    Buffer.add_trans(cp.deepcopy(trans1))
    Buffer.add_trans(cp.deepcopy(trans1))
    
    # test after fill up
    for i in range(N_CHOOSE):
        for j in range(len(Buffer.total_bufs[i])):
            print(str(i) + 'th action buff type '+str(Buffer.BUF_STRING[j]), Buffer.total_bufs[i][j])

    Buffer.shuffle_buffer(0)

    for i in range(N_CHOOSE):
        for j in range(len(Buffer.total_bufs[i])):
            print(str(i) + 'th action buff type '+str(Buffer.BUF_STRING[j]), Buffer.total_bufs[i][j])
    print('batch')
    print(Buffer.get_batch())

