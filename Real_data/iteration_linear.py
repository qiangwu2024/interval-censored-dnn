import numpy as np
from C_estimation import C_est
from I_spline import I_S
from g_linear import g_L

def Est_linear(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_test,X_sort1,X_sort0,Beta0,nodevec,m,c0):
    Lambda_U = I_S(m, c0, U_train, nodevec)
    Lambda_V = I_S(m, c0, V_train, nodevec)
    C_index = 0
    for loop in range(50):
        print('linear_iteration time=', loop)
        g_X = g_L(Z_train,X_train,De1_train,De2_train,De3_train,X_test,X_sort1,X_sort0,Lambda_U,Lambda_V)
        g_train = g_X['g_train']
        Beta1 = g_X['Beta']
        c1 = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,Z_train,Beta1,g_train,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Lambda_V = I_S(m,c1,V_train,nodevec)
        
        print('Beta=', Beta1)
        print('c=', c1)
        if (np.max(abs(Beta0-Beta1)) <= 0.001):
            C_index = 1
            break
        c0 = c1
        Beta0 = Beta1
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'g_test1': g_X['g_test1'],
        'g_test0': g_X['g_test0'],
        'c': c1,
        'Beta': Beta1,
        'C_index': C_index
    }




