import numpy as np
import tensorflow as tf
FLAGS = tf.flags.FLAGS

# NOTE: save the some utilities which to check collision or get the reward, use numpy for fast computation.  

def divide_xyr( matrix, transpose = False):
    """Divide the matrix (number of elements= N)*(x,y,z) into (N) * (x),N*y,N*r), return 3 array (N*x,N*y,N*r)
    """
    mat_x =np.reshape(matrix[:,0], (len(matrix),1))
    mat_y =np.reshape(matrix[:,1], (len(matrix),1))
    mat_r =np.reshape(matrix[:,2], (len(matrix),1))
    
    if transpose:
        mat_x = np.transpose(mat_x)
        mat_y = np.transpose(mat_y)
        mat_r = np.transpose(mat_r)
        
    return mat_x,mat_y,mat_r

def tranpose(vectors):
    """Input: some 1-D vectors
    
    Return: the transposed vectors
    """
    for i in range(len(vectors)):
        vectors[i]  = np.transpose(vectors[i])

    return vectors

def check_collision(matrix_a, matrix_b):
    """Get the two matrices with elements * (x,y,r)
    input: (matrix_a: N*(x,y,r),matrix_b:M*(x,y,r) )
    compare the distances between the elements and check the collisioned elements

    Return: 
    a_coll: index for collided elements in matrix_a if 0,2th row (or elements collide) a_coll = [0,2]
    b_coll: index for collided elements in matrix_b if 0,2th row (or elements collide) b_coll = [0,2]
    """
    a_x,a_y,a_r = divide_xyr(matrix_a)
    b_x,b_y,b_r = divide_xyr(matrix_b)

    ab_dist_sq = get_distance_matrix([a_x,a_y], [b_x,b_y])
    ab_radius_sq = get_radius_matrix(a_r,b_r)

    ## special case:when matrix_a, matrix_b is the same matrix
    if np.array_equal(matrix_a,matrix_b):
        ab_radius_sq = ab_radius_sq-np.diag(np.diag(ab_radius_sq))

    a_coll, b_coll = collsion_dist_radius(ab_dist_sq, ab_radius_sq)

    a_coll = np.squeeze(a_coll, axis=1)
    b_coll = np.squeeze(b_coll, axis=1)

    return a_coll.tolist(), b_coll.tolist()

def get_radius_matrix( a_r,b_r):
    """Generate the matrix R which entry is the sum of the radius 
    Input: two 1-dimensional vectors for a_r  = (ra1,ra2,...,raN), b_r = (rb1,rb2,...,rbM), with the radius

    Return: squre added matrix R with R[i,j] = (rai+rbj)^2  
    """
    b_r = np.transpose(b_r)

    a_b_r_sum = a_r + b_r
    a_b_r_sum_sq = np.multiply(a_b_r_sum, a_b_r_sum)

    return a_b_r_sum_sq

def collsion_dist_radius( dist_matrix,radius_matrix):
    """Input: 
    dist_matrix: N by M matrix which [i,j] entry denotes distance(ai,bj)^2
    radius_matrix: N by M matrix hich [i,j] entry denotes (radius(ai)+radius(bj))^2
    compare the distances between the elements from the distance matrix and the radius matrix
    
    Return: 
    a_coll: index for collided elements in matrix_a if 0,2th row (or elements collide) a_coll = [0,2]
    b_coll: index for collided elements in matrix_b if 0,2th row (or elements collide) b_coll = [0,2] 
    """
    # a_coll = []
    # b_coll = []

    # check whether the collisions occurs
    diff = dist_matrix - radius_matrix
    diff_a = np.amin(diff, axis = 1)
    diff_b = np.amin(diff, axis = 0)

    # collision list
    a_coll = np.argwhere(diff_a<0).astype(int)
    b_coll = np.argwhere(diff_b<0).astype(int)

    return a_coll, b_coll


def sum_circle_area(r, elements):
    """r: array of the radius of the circles:, r[0]:0th circle's radius
    elements: selected circles to be added

    Return: summation of area of the selected circles
    """
    area =   np.pi * np.sum(np.multiply(r[elements], r[elements]))
    
    return area

def get_distance_matrix(a_xy, b_xy):
    """calculate the distance matrix between the matrixes 
    a_xy = [[a_x[0],a_x[1],...,a_x[N]],[a_y[0],a_y[1],...,a_y[N]]]
    b_xy = [[b_x[0],b_x[1],...,b_x[M]],[b_y[0],b_y[1],...,b_y[M]]]

    Return the N by M matrix: a_b_dis_sq with  a_b_dis_sq[i,j]:= (distance(a_x[i],a_y[i]), (b_x[j],b_y[j])^2)
    """
    ## divide the vectors to using broadcast
    a_x,a_y = a_xy
    b_x,b_y = tranpose(b_xy)

    ## using (a-b)^2 = (a+b)^2 - 4ab
    # get (a+b)^2
    a_b_x_sum =  a_x + b_x
    a_b_y_sum =  a_y + b_y
    a_b_x_sq = np.multiply(a_b_x_sum, a_b_x_sum) 
    a_b_y_sq = np.multiply(a_b_y_sum, a_b_y_sum) 

    # get ab
    a_b_x_mul = np.matmul(a_x, b_x)
    a_b_y_mul = np.matmul(a_y, b_y)

    # get distance by get (a+b)^2 - 4ab
    a_b_dis_sq = a_b_x_sq + a_b_y_sq - 4 * (a_b_x_mul + a_b_y_mul)

    return a_b_dis_sq

def generate_xy_elements(Number_elements):
    """Goal: randomly generate the features (x,y,r) for the newly generated elements
    third generated feature 3-dimensional r does not used in the outside code (dummy but for fast computation)
    
    Number_elements: denotes the number of the elements should be generated

    Return: the matrix for Number_elements *(x,y,r)s
    """
    xy_matrix = 2 * FLAGS.UB_CENTER * (np.random.rand(Number_elements, 3) - 0.5 * np.ones([Number_elements, 3]))
    
    return xy_matrix

def generate_r_elements( Number_elements, small_start = False):
    """[Goal] randomly generate the radius r of for the newly generated elements or circles
    Number_elements: number of the elements that would be newly generated 
    small_start is true (when collision occurs), generate very small radius 
    
    Return: the matrix for Number_elements*(r)s
    """
    r_matrix = FLAGS.MAX_RADIUS* np.random.rand(Number_elements)
    if small_start:
        r_matrix = FLAGS.INIT_RADIUS * r_matrix

    return r_matrix

if __name__ == '__main__':
    a = np.array([[-0.43418046, -0.36410117,  0.09058379]])
    b = np.array([[-0.22148598, -0.16354707,  0.08652431]])
    print(check_collision(a,b))