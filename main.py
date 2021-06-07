import numpy as np

import time
import search
import mining


some_2d_underground_1 = np.array([
       [-0.814,  0.637, 1.824, -0.563],
       [ 0.559, -0.234, -0.366,  0.07 ],
       [ 0.175, -0.284,  0.026, -0.316],
       [ 0.212,  0.088,  0.304,  0.604],
       [-1.231, 1.558, -0.467, -0.371]])

some_3d_underground_1 = np.array([[[ 0.455,  0.579, -0.54 , -0.995, -0.771],
                                   [ 0.049,  1.311, -0.061,  0.185, -1.959],
                                   [ 2.38 , -1.404,  1.518, -0.856,  0.658],
                                   [ 0.515, -0.236, -0.466, -1.241, -0.354]],
                                  [[ 0.801,  0.072, -2.183,  0.858, -1.504],
                                   [-0.09 , -1.191, -1.083,  0.78 , -0.763],
                                   [-1.815, -0.839,  0.457, -1.029,  0.915],
                                   [ 0.708, -0.227,  0.874,  1.563, -2.284]],
                                  [[ -0.857,  0.309, -1.623,  0.364,  0.097],
                                   [-0.876,  1.188, -0.16 ,  0.888, -0.546],
                                   [-1.936, -3.055, -0.535, -1.561, -1.992],
                                   [ 0.316,  0.97 ,  1.097,  0.234, -0.296]]])


def test_2D_search_dig_plan():
    # x_len, z_len = 5,4
    # some_neg_bias = -0.2
    # my_underground = np.random.randn(x_len, z_len) + some_neg_bias
    
    my_underground = some_2d_underground_1

    mine = mining.Mine(my_underground)   
    mine.console_display()
    
    # print(my_underground.__repr__())
    
    print('-------------- BB computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = mining.search_bb_dig_plan(mine)
    toc = time.time() 
    print('BB Best payoff ',best_payoff)
    print('BB Best final state ', best_final_state)  
    print('BB action list ', best_a_list)
    print('BB Computation took {} seconds'.format(toc-tic))  

    sanity_matches = np.array_equal(best_a_list, [(0,), (1,), (2,), (3,), (4,), (0,), (1,), (2,), (3,), (4,), (0,), (2,), (3,), (4,), (3,)])
    print('Action list matches sanity test: {}\n'.format(sanity_matches))
    assert sanity_matches

    print('-------------- DP computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = mining.search_dp_dig_plan(mine)
    toc = time.time() 
    print('DP Best payoff ',best_payoff)
    print('DP Best final state ', best_final_state)  
    print('DP action list ', best_a_list)
    print('DP Computation took {} seconds'.format(toc-tic))  
    sanity_matches = np.array_equal(best_a_list, ((0,), (1,), (0,), (2,), (1,), (0,), (3,), (2,), (4,), (3,), (2,), (4,), (3,), (4,), (3,)))
    print('Action list matches sanity test: {}\n'.format(sanity_matches))
    assert sanity_matches

def test_3D_search_dig_plan():
    # np.random.seed(10)

    # x_len,y_len,z_len = 3,4,5
    # some_neg_bias = -0.3    
    # my_underground = np.random.randn(x_len,y_len,z_len) + some_neg_bias
    
    my_underground =  some_3d_underground_1
    
    mine = mining.Mine(my_underground)   
    mine.console_display()

    print('-------------- BB computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = mining.search_bb_dig_plan(mine)
    toc = time.time() 
    print('BB Best payoff ',best_payoff)
    print('BB Best final state ', best_final_state)  
    print('BB action list ', best_a_list)
    print('BB Computation took {} seconds'.format(toc-tic))  

    sanity_matches = np.array_equal(best_a_list, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 3), (0, 0)])
    print('Action list matches sanity test: {}\n'.format(sanity_matches))
    assert sanity_matches


    print('-------------- DP computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = mining.search_dp_dig_plan(mine)
    toc = time.time() 
    print('DP Best payoff ',best_payoff)
    print('DP Best final state ', best_final_state)  
    print('DP action list ', best_a_list)
    print('DP Computation took {} seconds'.format(toc-tic))

    sanity_matches = np.array_equal(best_a_list, ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (0, 0), (1, 3), (2, 3)))
    print('Action list matches sanity test: {}\n'.format(sanity_matches))
    assert sanity_matches

def main():

    print('= '*20)
    test_2D_search_dig_plan()
    print('= '*20)
    test_3D_search_dig_plan()


if __name__ == "__main__":
    main()