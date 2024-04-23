#%% ----------------------
import numpy as np
import torch
import tensorflow as tf
import pandas as pd
import time
from data_generator import generate_scenario_4
from iteration_deep import Est_deep
from I_spline import I_S
from iteration_NN_IC import Est_NN_IC
from Bernstein_Poly import Bern_S
#%% ---------------------
def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    tf.random.set_seed(seed) 

set_seed(1) 

#%% -----------------------
n = 1000 
p_dim_list = np.array([20, 50])
lambda_param = 0.01
k_param = 10
tau = 100
B = 200
tt = np.array(np.linspace(tau/500, tau, 500), dtype="float32")

n_layer = 3
n_node_list = np.array([50, 75])
n_epoch = 1000
n_lr = 1e-3
p = 3
m = 8
nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32")
c0 = np.array(0.1*np.ones(m+p), dtype="float32")

L_1_penalty = 0.5
N_node = 50
N_epoch = 1000
N_lr = 0.01
m_n = 3

def generate_increasing_random_numbers(n):
    numbers = []
    prev_number = 0.0
    for _ in range(n):
        number = np.random.uniform(prev_number, 1)
        numbers.append(number)
        prev_number = number
    return numbers

b0 = np.array(generate_increasing_random_numbers(m_n+1), dtype="float32")

run_time_D_all = []; run_time_NN_IC_all = [] 
mspe_D_all = []; sse_D_all = []; mspe_NN_IC_all = []; sse_NN_IC_all = []
for k in range(len(p_dim_list)):
    p_dim = p_dim_list[k]
    n_node = n_node_list[k]
    Mspe_D = []; Mspe_NN_IC=[]
    run_time_all_D = 0; run_time_all_NN_IC = 0
    for b in range(B):
        print('---p=', p_dim,'---b=', b)
        #%% Training parameters
        train_data = generate_scenario_4(n, p_dim, lambda_param, k_param, tau)
        Z_train = train_data['Z']
        U_train = train_data['U']
        V_train = train_data['V']
        De1_train = train_data['De1']
        De2_train = train_data['De2']
        De3_train = train_data['De3']
        g_train = train_data['g_Z']

        #%% validation data
        validation_data = generate_scenario_4(200, p_dim, lambda_param, k_param, tau)

        #%% test data
        test_data = generate_scenario_4(200, p_dim, lambda_param, k_param, tau)
        Z_test = test_data['Z']   
        g_true = test_data['g_Z']
        Lambda_true = (tt*lambda_param)**k_param
        S_true = np.exp(-np.dot(np.exp(np.reshape(g_true, (len(g_true), 1))), np.reshape(Lambda_true, (1, len(Lambda_true)))))
        
        #%% Our Method
        start_time_D = time.time()
        Est_D = Est_deep(train_data,validation_data,Z_test,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0,lambda_param,k_param)
        Lambda_D = I_S(m,Est_D['c'],tt,nodevec)
        S_D = np.exp(-np.dot(np.exp(np.reshape(Est_D['g_test'], (len(Est_D['g_test']), 1))), np.reshape(Lambda_D, (1, len(Lambda_D)))))
        Mspe_D.append(np.mean((S_D-S_true)**2))
        end_time_D = time.time()

        run_time_D = end_time_D - start_time_D
        run_time_all_D  += run_time_D
        if (b == B-1):
            run_time_D_all.append(run_time_all_D)
            print('run_time_all_D=', run_time_all_D, 'seconds')
    
        #%% NN_IC Method
        start_time_NN_IC = time.time()
        est_NN_IC = Est_NN_IC(train_data,validation_data,Z_test,L_1_penalty,N_node,N_lr,N_epoch,m_n,b0,tau)
        Lambda_NN_IC = np.dot(Bern_S(m_n, tt, 0, tau), np.array(est_NN_IC['B1']))
        S_NN_IC = np.exp(-np.dot(np.exp(np.reshape(est_NN_IC['g_test'], (len(est_NN_IC['g_test']), 1))), np.reshape(Lambda_NN_IC, (1, len(Lambda_NN_IC))))) # 200*1000
        Mspe_NN_IC.append(np.mean((S_NN_IC-S_true)**2))
        end_time_NN_IC = time.time()

        run_time_NN_IC = end_time_NN_IC - start_time_NN_IC
        run_time_all_NN_IC  += run_time_NN_IC
        if (b == B-1):
            run_time_NN_IC_all.append(run_time_all_NN_IC )
            print('run_time_all_NN_IC=', run_time_all_NN_IC, 'seconds')
    
    
    #%% Mean Squared Prediction Error
    # Our Method
    Mspe_D_mean = np.mean(np.array(Mspe_D))
    Sse_D = np.sqrt(np.mean((np.array(Mspe_D)-Mspe_D_mean)**2))
    mspe_D_all.append(Mspe_D_mean)
    sse_D_all.append(Sse_D)
    # NN_IC Method
    Mspe_NN_IC_mean = np.mean(np.array(Mspe_NN_IC))
    Sse_NN_IC = np.sqrt(np.mean((np.array(Mspe_NN_IC)-Mspe_NN_IC_mean)**2))
    mspe_NN_IC_all.append(Mspe_NN_IC_mean)
    sse_NN_IC_all.append(Sse_NN_IC)
    print('Mspe_D_mean =', Mspe_D_mean)
    print('Sse_D =', Sse_D)
    print('Mspe_NN_IC_mean =', Mspe_NN_IC_mean)
    print('Sse_NN_IC =', Sse_NN_IC)


#%% -------save results--------
dic_mspe = {"p": p_dim_list, "mspe_D_all": np.array(mspe_D_all), "sse_D_all": np.array(sse_D_all), "run_time_D_all": np.array(run_time_D_all), "mspe_NN_IC_all": np.array(mspe_NN_IC_all), "sse_NN_IC_all": np.array(sse_NN_IC_all), "run_time_NN_IC_all": np.array(run_time_NN_IC_all)}
result_mspe = pd.DataFrame(dic_mspe)
result_mspe.to_csv('result_scenario4.csv')
run_time_D_all