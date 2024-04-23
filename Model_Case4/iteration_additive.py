import numpy as np
from C_estimation import C_est
from I_spline import I_S
from g_additive import g_A

def Est_additive(train_data,X_test,Beta0,nodevec,m,c0,m0,nodevec0):
    Z_train = train_data['Z']
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']
    Beta0 = np.array([Beta0])
    Lambda_U = I_S(m, c0, U_train, nodevec)
    Lambda_V = I_S(m, c0, V_train, nodevec)
    C_index = 0
    for loop in range(30):
        print('additive_iteration time=', loop)
        g_X = g_A(train_data,X_test,Lambda_U,Lambda_V,m0,nodevec0)
        g_train = g_X['g_train']
        Beta1 = g_X['beta']
        c1 = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,Z_train,Beta0,g_train,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Lambda_V = I_S(m,c1,V_train,nodevec)
        print('Beta=', Beta1)
        print('c=', c1)
        if (abs(Beta0-Beta1) <= 0.001):
            C_index = 1
            break
        c0 = c1
        Beta0 = Beta1
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'c': c1,
        'Beta': Beta1,
        'C_index': C_index
    }





