import numpy as np
from C_estimation import C_est
from I_spline import I_S
from g_deep import g_D

def Est_deep(train_data,validation_data,Z_test,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0,lambda_param,k_param):
    # Z_train = train_data['Z']
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']

    Lambda_U = I_S(m,c0,U_train,nodevec)
    Lambda_V = I_S(m,c0,V_train,nodevec)
    C_index = 0

    loss_validation_0 = 0
    loss_validation = 0
    for loop in range(100):
        print('deep_iteration time=', loop)
        if (loss_validation > loss_validation_0) and (loop >= 5):
            C_index = 1
            break
        else:
            loss_validation_0 = loss_validation
        g_Z = g_D(train_data,validation_data['Z'],Z_test,Lambda_U,Lambda_V,n_layer,n_node,n_lr,n_epoch,lambda_param,k_param)
        g_train = g_Z['g_train']
        g_validation = g_Z['g_validation']
        c1 = C_est(m,U_train,V_train,De1_train,De2_train,De3_train,g_train,nodevec)
        Lambda_U = I_S(m,c1,U_train,nodevec)
        Lambda_V = I_S(m,c1,V_train,nodevec)
        Lambda_U_validation = I_S(m,c1,validation_data['U'],nodevec)
        Lambda_V_validation = I_S(m,c1,validation_data['V'],nodevec)
        
        loss_validation = - np.mean(validation_data['De1'] * np.log(1 - np.exp(- Lambda_U_validation * np.exp(g_validation)) + 1e-5) + validation_data['De2'] * np.log(np.exp(- Lambda_U_validation * np.exp(g_validation)) - np.exp(- Lambda_V_validation * np.exp(g_validation)) + 1e-5) - validation_data['De3'] * Lambda_V_validation * np.exp(g_validation))
        c0 = c1
    
    return {
        'g_train': g_train,
        'g_test': g_Z['g_test'],
        'c': c1,
        'C_index': C_index
    }
