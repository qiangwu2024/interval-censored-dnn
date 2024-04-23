import numpy as np
from Beta_estimate import Beta_est
from C_estimation import C_est
from I_spline import I_S
from g_deep_judge import g_D_judge

def Est_deep_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_ones1,X_ones2,Beta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0):
    Lambda_U = I_S(m,c0,U_train,nodevec)
    Lambda_V = I_S(m,c0,V_train,nodevec)
    C_index = 0
    for loop in range(50):
        print('deep_iteration time=', loop)
        g_X = g_D_judge(Z_train,X_train,De1_train,De2_train,De3_train,X_ones1,X_ones2,Lambda_U,Lambda_V,Beta0,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']
        c1 = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,Z_train,Beta0,g_train,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Lambda_V = I_S(m,c1,V_train,nodevec)
        Beta1 = Beta_est(De1_train,De2_train,De3_train,Z_train,Lambda_U,Lambda_V,g_train)
        print('Beta=', Beta1)
        print('c=', c1)
        if (np.max(abs(Beta0-Beta1)) <= 0.001):
            C_index = 1
            break
        c0 = c1
        Beta0 = Beta1
    
    return {
        'g_train': g_train,
        'g_value1': g_X['g_value1'],
        'g_value2': g_X['g_value2'],
        'c': c1,
        'Beta': Beta1,
        'C_index': C_index
    }
