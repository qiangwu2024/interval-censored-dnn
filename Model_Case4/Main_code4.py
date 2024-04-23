#%% ---------------------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import generate_case_4
from iteration_deep import Est_deep
from iteration_linear import Est_linear
from iteration_additive import Est_additive
from I_spline import I_S
from Least_FD import LFD

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) 

set_seed(8)

tau = 20
p = 3
Set_n = np.array([1000, 2000])
corr = 0.5 
n_layer = 3 
n_node = 50 
n_epoch = 200 
Set_lr = np.array([3e-4, 4e-4])
Beta = 1 

node_D = np.array([45, 50]) 
lr_D = np.array([3e-4, 1e-3])

node_L = np.array([40, 40])
lr_L = np.array([3e-4, 3e-4])

node_A = np.array([50, 50]) 
lr_A = np.array([2e-5, 2e-5])

B = 200

test_data = generate_case_4(200, corr, Beta, tau)
X_test = test_data['X']
g_true = test_data['g_X']
dim_x = X_test.shape[0] 
u_value = np.array(np.linspace(0, tau, 50), dtype="float32")
Lambda_true = np.sqrt(u_value)*4 / 17
m = 10
nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32")
m0 = 4
nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")


#%% ----------------Main results------------------------
fig1 = plt.figure()
ax1_1 = fig1.add_subplot(1, 2, 1)
plt.ylim(-2,2) 
ax1_1.set_title("Case 4, n=1000",fontsize=10)
ax1_1.set_xlabel("Predictor",fontsize=8) 
ax1_1.set_ylabel("Error",fontsize=8) 
ax1_1.tick_params(axis='both',labelsize=6) 

ax1_2 = fig1.add_subplot(1, 2, 2)
plt.ylim(-2,2) 
ax1_2.set_title("Case 4, n=2000",fontsize=10) 
ax1_2.set_xlabel("Predictor",fontsize=8) 

ax1_2.tick_params(axis='both',labelsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)



fig2 = plt.figure()
ax2_1 = fig2.add_subplot(1, 2, 1)
plt.ylim(0,1.5) 
ax2_1.set_title("Case 4, n=1000", fontsize=10)
ax2_1.set_xlabel("Time",fontsize=8) 
ax2_1.set_ylabel("Cumulative hazard function",fontsize=8) 
ax2_1.tick_params(axis='both',labelsize=6) 
ax2_1.plot(u_value, Lambda_true, color='k', label='True')
ax2_1.legend(loc='upper left', fontsize=6)

ax2_2 = fig2.add_subplot(1, 2, 2)
plt.ylim(0,1.5) 
ax2_2.set_title("Case 4, n=2000", fontsize=10) 
ax2_2.set_xlabel("Time",fontsize=8) 
ax2_2.tick_params(axis='both',labelsize=6) 
ax2_2.plot(u_value, Lambda_true, color='k', label='True')
ax2_2.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)


# --- Save Bias, SSE, ESE, CP---
Bias_D = []; Sse_D = []; Ese_D = []; Cp_D = []; Re_D = []; G_D_sd = []
Bias_L = []; Sse_L = []; Ese_L = []; Cp_L = []; Re_L = []; G_L_sd = []
Bias_A = []; Sse_A = []; Ese_A = []; Cp_A = []; Re_A = []; G_A_sd = []

for i in range(len(Set_n)):
    n = Set_n[i]
    n_lr = Set_lr[i]
    #%% ------------ Store the results of B loops ---------
    G_test_D = []; C_D=[]; beta_D = []; Info_D = []; re_D = []
    G_test_L = []; C_L = []; beta_L = []; Info_L = []; re_L = []
    G_test_A = []; C_A = []; beta_A = []; Info_A = []; re_A = []
    for b in range(B):
        print('n=', n, 'b=', b)
        set_seed(12 + b)
        #%% ------------------------
        c0 = np.array(0.1*np.ones(m+p), dtype="float32")
        Beta0 = np.array(0, dtype='float32')
        #%% -----------------------
        # train data
        train_data = generate_case_4(n, corr, Beta, tau)
        Z_train = train_data['Z']
        U_train = train_data['U']
        V_train = train_data['V']
        De1_train = train_data['De1']
        De2_train = train_data['De2']
        De3_train = train_data['De3']
        g_train = train_data['g_X']
        #%% ==============Deep method==============
        Est_D = Est_deep(train_data=train_data,X_test=X_test,Beta=Beta,Beta0=Beta0,n_layer=n_layer,n_node=n_node,n_lr=n_lr,n_epoch=n_epoch,nodevec=nodevec,m=m,c0=c0)
        G_test_D.append(Est_D['g_test'])
        re_D.append(np.sqrt(np.mean((Est_D['g_test']-np.mean(Est_D['g_test'])-g_true)**2)/np.mean(g_true**2)))
        C_D.append(Est_D['c'])
        a_b_D = LFD(train_data,I_S(m,Est_D['c'],U_train,nodevec),I_S(m,Est_D['c'],V_train,nodevec),Est_D['g_train'],Est_D['Beta'],n_layer,n_node=node_D[i],n_lr=lr_D[i],n_epoch=200)
        f_U = I_S(m,Est_D['c'],U_train,nodevec) * np.exp( Z_train * Est_D['Beta'] + Est_D['g_train'])
        f_V = I_S(m,Est_D['c'],V_train,nodevec) * np.exp( Z_train * Est_D['Beta'] + Est_D['g_train'])
        Ef_U = np.exp(- f_U)
        Ef_V = np.exp(- f_V)
        Q1_y = Ef_U/(1-Ef_U)
        Q2_y = Ef_U/(Ef_U-Ef_V)
        Q3_y = Ef_V/(Ef_U-Ef_V)
        Es_D = f_U*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train-a_b_D[:,0]) + f_V*(De2_train*Q3_y-De3_train)*(Z_train-a_b_D[:,1])
        beta_D.append(Est_D['Beta'])
        if np.isnan(np.mean(Es_D**2)):
            if (i == 0):
                Info_D.append(0.25)
            else:
                Info_D.append(0.4)
        else:
            Info_D.append(np.mean(Es_D**2))
        print('Info_D=', np.mean(Es_D**2))
        print('Info_D=', Info_D)
        
        #%% =============Linear method============
        Est_L = Est_linear(train_data,X_test,Beta0,nodevec,m,c0)
        G_test_L.append(Est_L['g_test'])
        re_L.append(np.sqrt(np.mean((Est_L['g_test']-np.mean(Est_L['g_test'])-g_true)**2)/np.mean(g_true**2)))
        C_L.append(Est_L['c'])
        a_b_L = LFD(train_data,I_S(m,Est_L['c'],U_train,nodevec),I_S(m,Est_L['c'],V_train,nodevec),Est_L['g_train'],Est_L['Beta'],n_layer,n_node=node_L[i],n_lr=lr_L[i],n_epoch=200)
        f_U_L = I_S(m,Est_L['c'],U_train,nodevec) * np.exp( Z_train * Est_L['Beta'] + Est_L['g_train'])
        f_V_L = I_S(m,Est_L['c'],V_train,nodevec) * np.exp( Z_train * Est_L['Beta'] + Est_L['g_train'])
        Ef_U_L = np.exp(- f_U_L)
        Ef_V_L = np.exp(- f_V_L)
        Q1_y = Ef_U_L/(1-Ef_U_L)
        Q2_y = Ef_U_L/(Ef_U_L-Ef_V_L)
        Q3_y = Ef_V_L/(Ef_U_L-Ef_V_L)
        Es_L = f_U_L*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train-a_b_L[:,0]) + f_V_L*(De2_train*Q3_y-De3_train)*(Z_train-a_b_L[:,1])
        beta_L.append(Est_L['Beta'])
        if np.isnan(np.mean(Es_L**2)):
            if (i == 0):
                Info_L.append(0.25)
            else:
                Info_L.append(0.4)
        else:
            Info_L.append(np.mean(Es_L**2))
        print('Info_L=', np.mean(Es_L**2))
        print('Info_L=', Info_L)

        #%% ============PLACM method==============
        Est_A = Est_additive(train_data,X_test,Beta0,nodevec,m,c0,m0,nodevec0)
        G_test_A.append(Est_A['g_test'])
        re_A.append(np.sqrt(np.mean((Est_A['g_test']-np.mean(Est_A['g_test'])-g_true)**2)/np.mean(g_true**2)))
        C_A.append(Est_A['c'])
        a_b_A = LFD(train_data,I_S(m,Est_A['c'],U_train,nodevec),I_S(m,Est_A['c'],V_train,nodevec),Est_A['g_train'],Est_A['Beta'],n_layer,n_node=node_A[i],n_lr=lr_A[i],n_epoch=200)
        f_U_A = I_S(m,Est_A['c'],U_train,nodevec) * np.exp( Z_train * Est_A['Beta'] + Est_A['g_train'])
        f_V_A = I_S(m,Est_A['c'],V_train,nodevec) * np.exp( Z_train * Est_A['Beta'] + Est_A['g_train'])
        Ef_U_A = np.exp(- f_U_A)
        Ef_V_A = np.exp(- f_V_A)
        Q1_y = Ef_U_A/(1-Ef_U_A)
        Ef_U_V_A = Ef_U_A-Ef_V_A
        Q2_y = Ef_U_A/Ef_U_V_A
        Q3_y = Ef_V_A/Ef_U_V_A
        Es_A = f_U_A*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train-a_b_A[:,0]) + f_V_A*(De2_train*Q3_y-De3_train)*(Z_train-a_b_A[:,1])
        beta_A.append(Est_A['Beta'])
        if np.isnan(np.mean(Es_A**2)):
            if (i == 0):
                Info_A.append(0.25)
            else:
                Info_A.append(0.4)
        else:
            Info_A.append(np.mean(Es_A**2))
        print('Info_A=', np.mean(Es_A**2))
        print('Info_A=', Info_A)
        
        
    #%% ============Figures for DNN-based================
    Error_D = np.mean(np.array(G_test_D), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_D, s=4, marker='o', label='DNN-based')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_D), axis=0),u_value,nodevec), label='DNN-based', linestyle='--')
        ax2_1.legend(loc='upper left', fontsize=6)
    else:
        ax1_2.scatter(np.arange(dim_x), Error_D, s=4, marker='o', label='DNN-based')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_D), axis=0),u_value,nodevec), label='DNN-based', linestyle='--')
        ax2_2.legend(loc='upper left', fontsize=6)
    # --------- Bias_D, SSE_D, ESE_D, CP_D of hat_Beta for Deep-----------
    Bias_D.append(np.mean(np.array(beta_D))-Beta)
    Sse_D.append(np.sqrt(np.mean((np.array(beta_D)-np.mean(np.array(beta_D)))**2)))
    Ese_D.append(1/np.sqrt(n*np.mean(np.array(Info_D))))
    Cp_D.append(np.mean((np.array(beta_D)-1.96/np.sqrt(n*np.mean(np.array(Info_D)))<=Beta)*(Beta<=np.array(beta_D)+1.96/np.sqrt(n*np.mean(np.array(Info_D))))))
    # ----- relative error and standard deviation of hat_g for DNN-based -----
    Re_D.append(np.mean(re_D))
    G_D_sd.append(np.sqrt(np.mean((re_D-np.mean(re_D))**2)))
    #%% ============ Figures for CPH ===================
    Error_L = np.mean(np.array(G_test_L), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_1.legend(loc='upper left', fontsize=6)
    else:
        ax1_2.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_2.legend(loc='upper left', fontsize=6)
    # --------- Bias_L, SSE_L, ESE_L, CP_L of hat_Beta for CPH -----------
    Bias_L.append(np.mean(np.array(beta_L))-Beta)
    Sse_L.append(np.sqrt(np.mean((np.array(beta_L)-np.mean(np.array(beta_L)))**2)))
    Ese_L.append(1/np.sqrt(n*np.mean(np.array(Info_L))))
    Cp_L.append(np.mean((np.array(beta_L)-1.96/np.sqrt(n*np.mean(np.array(Info_L)))<=Beta)*(Beta<=np.array(beta_L)+1.96/np.sqrt(n*np.mean(np.array(Info_L))))))
    # ----- relative error and standard deviation of hat_g  for CPH -----
    Re_L.append(np.mean(re_L))
    G_L_sd.append(np.sqrt(np.mean((re_L-np.mean(re_L))**2)))
    
    # #%% ============ Figures for PLACM ===================
    Error_A = np.mean(np.array(G_test_A), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_1.legend(loc='upper left', fontsize=6)
    else:
        ax1_2.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_2.legend(loc='upper left', fontsize=6)
    # --------- Bias_A, SSE_A, ESE_A, CP_A of hat_Beta for PLACM -----------
    Bias_A.append(np.mean(np.array(beta_A))-Beta)
    Sse_A.append(np.sqrt(np.mean((np.array(beta_A)-np.mean(np.array(beta_A)))**2)))
    Ese_A.append(1/np.sqrt(n*np.mean(np.array(Info_A))))
    Cp_A.append(np.mean((np.array(beta_A)-1.96/np.sqrt(n*np.mean(np.array(Info_A)))<=Beta)*(Beta<=np.array(beta_A)+1.96/np.sqrt(n*np.mean(np.array(Info_A))))))
    # ----- relative error and standard deviation of hat_g  for PLACM -----
    Re_A.append(np.mean(re_A))
    G_A_sd.append(np.sqrt(np.mean((re_A-np.mean(re_A))**2)))
    
    
#%% -----------Save all results------------
# ================figures======================
fig1.savefig('fig_g_4_t.jpeg', dpi=400, bbox_inches='tight')
fig2.savefig('fig_Lambda_4_t.jpeg', dpi=400, bbox_inches='tight')

# =================tables=======================
dic_error = {"n": Set_n, "Bias_deep": np.array(Bias_D), "SSE_deep": np.array(Sse_D), "ESE_deep": np.array(Ese_D), "CP_deep": np.array(Cp_D), "Bias_L": np.array(Bias_L),  "SSE_L": np.array(Sse_L), "ESE_L": np.array(Ese_L), "CP_L": np.array(Cp_L), "Bias_A": np.array(Bias_A), "SSE_A": np.array(Sse_A), "ESE_A": np.array(Ese_A), "CP_A": np.array(Cp_A)}
result_error = pd.DataFrame(dic_error)
result_error.to_csv('result_error_deep2.csv')

dic_re = {"n": Set_n, "Re_deep": np.array(Re_D), "G_deep_sd": np.array(G_D_sd), "Re_L": np.array(Re_L), "G_L_sd": np.array(G_L_sd), "Re_A": np.array(Re_A), "G_A_sd": np.array(G_A_sd)}
result_re = pd.DataFrame(dic_re)
result_re.to_csv('result_re_deep2.csv')