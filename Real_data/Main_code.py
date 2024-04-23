#%% -------------------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from I_spline import I_S
from Least_FD import LFD
from iteration_deep_judge import Est_deep_judge
from iteration_linear_judge import Est_linear_judge
from iteration_additive_judge import Est_additive_judge
from iteration_deep import Est_deep
from iteration_linear import Est_linear
from iteration_additive import Est_additive

#%% ----------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(8)
#%% --------------------
p = 3
n_layer = 3
n_node = 150
n_epoch = 1000
n_lr = 1e-3
m = 16

#%% Data Processing
df = pd.read_csv('data_center.csv')

Z = np.array(df[["BMI01","GLUCOS01","HDL01","TCHSIU01","V1AGE01"]], dtype='float32')
X = np.array(df[["SBPA21","SBPA22","RACEGRP","GENDER","Cen1","Cen2","Cen3","SBPA21_real","SBPA22_real","V1AGE01_real"]], dtype='float32')
U = np.array(df["U"], dtype='float32')
V = np.array(df["V"], dtype='float32')
De1 = np.array(df["De1"], dtype='float32')
De2 = np.array(df["De2"], dtype='float32')
De3 = np.array(df["De3"], dtype='float32')

nodevec = np.array(np.linspace(0, 10200, m+2), dtype="float32")
c0 = np.array(0.5*np.ones(m+p), dtype="float32")
Beta0 = np.array(0.5*np.ones(Z.shape[1]), dtype='float32')

m0 = 4
nodevec0 = np.array(np.linspace(-10, 10, m0+2), dtype="float32")
#%% ===========================
A = np.arange(len(U))
np.random.shuffle(A)

Z_R = Z[A]
X_R = X[A]
U_R = U[A]
V_R = V[A]
De1_R = De1[A]
De2_R = De2[A]
De3_R = De3[A]

# ---train data 11000
Z_R_train = Z_R[np.arange(11000)]
X_train_all = X_R[np.arange(11000)]
X_R_train = X_train_all[:,0:7]
U_R_train = U_R[np.arange(11000)]
V_R_train = V_R[np.arange(11000)]
De1_R_train = De1_R[np.arange(11000)]
De2_R_train = De2_R[np.arange(11000)]
De3_R_train = De3_R[np.arange(11000)]
# ---test data 1204
Z_R_test = Z_R[np.arange(11000,len(U))]
X_R_test = X_R[np.arange(11000,len(U))][:,0:7]
U_R_test = U_R[np.arange(11000,len(U))]
V_R_test = V_R[np.arange(11000,len(U))]
De1_R_test = De1_R[np.arange(11000,len(U))]
De2_R_test = De2_R[np.arange(11000,len(U))]
De3_R_test = De3_R[np.arange(11000,len(U))]


#%% Divide the samples of the test set into two classes（delta3=1, delta3=0）
# --------- delta3 = 1----------
U_test1 = np.array(U_R_test[De3_R_test==1])
V_test1 = np.array(V_R_test[De3_R_test==1])
Z_test1 = np.array(Z_R_test[De3_R_test==1])
X_test1 = np.array(X_R_test[De3_R_test==1])

X_sort1 = X_test1[V_test1.argsort()]
Z_sort1 = Z_test1[V_test1.argsort()]
U_sort1 = U_test1[V_test1.argsort()]
V_sort1 = V_test1[V_test1.argsort()]

n_V1 = len(V_sort1)
V1_015 = V_sort1[round(n_V1*0.15)]
V1_030 = V_sort1[round(n_V1*0.3)]
V1_045 = V_sort1[round(n_V1*0.45)]
V1_060 = V_sort1[round(n_V1*0.6)]
V1_075 = V_sort1[round(n_V1*0.75)]
V1 = [V1_015, V1_030, V1_045, V1_060, V1_075]

Z_sort1_subject = Z_sort1[[round(n_V1*0.15),round(n_V1*0.3),round(n_V1*0.45),round(n_V1*0.6),round(n_V1*0.75)]]
X_sort1_subject = X_sort1[[round(n_V1*0.15),round(n_V1*0.3),round(n_V1*0.45),round(n_V1*0.6),round(n_V1*0.75)]]

V1_value = np.array(np.linspace(0, 10200, 30), dtype="float32")

# --------- delta3 = 0----------
U_test0 = np.array(U_R_test[De3_R_test==0])
V_test0 = np.array(V_R_test[De3_R_test==0])
Z_test0 = np.array(Z_R_test[De3_R_test==0])
X_test0 = np.array(X_R_test[De3_R_test==0])

X_sort0 = X_test0[V_test0.argsort()]
Z_sort0 = Z_test0[V_test0.argsort()]
U_sort0 = U_test0[V_test0.argsort()]
V_sort0 = V_test0[V_test0.argsort()]

n_V0 = len(V_sort0)
V0_015 = V_sort0[round(n_V0*0.15)]
V0_030 = V_sort0[round(n_V0*0.3)]
V0_045 = V_sort0[round(n_V0*0.45)]
V0_060 = V_sort0[round(n_V0*0.6)]
V0_075 = V_sort0[round(n_V0*0.75)]

V0 = [V0_015, V0_030, V0_045, V0_060, V0_075]

Z_sort0_subject = Z_sort0[[round(n_V0*0.15),round(n_V0*0.3),round(n_V0*0.45),round(n_V0*0.6),round(n_V0*0.75)]]
X_sort0_subject = X_sort0[[round(n_V0*0.15),round(n_V0*0.3),round(n_V0*0.45),round(n_V0*0.6),round(n_V0*0.75)]]
V0_value = np.array(np.linspace(0, 10200, 30), dtype="float32")


#%% =========================================
X_class = X_train_all[(X_train_all[:,2]==0)*(X_train_all[:,3]==1)*(X_train_all[:,4]==0)*(X_train_all[:,5]==0)*(X_train_all[:,6]==1)]

X_mean = np.mean(X_class[:,0:7], axis=0)
X_data1 = np.tile(X_mean, (100,1))
Min1 = np.min(X_class[:,0])
Max1 = np.max(X_class[:,0])
x_value1 = np.array(np.linspace(Min1, Max1, 100), dtype="float32")
X_data1[:,0] = x_value1

X1_min = np.min(X_class[:,7])
X1_max = np.max(X_class[:,7])
X_value1 = np.array(np.linspace(X1_min, X1_max, 100), dtype="float32")


X_data2 = np.tile(X_mean, (100,1))
Min2 = np.min(X_class[:,1])
Max2 = np.max(X_class[:,1])
x_value2 = np.array(np.linspace(Min2, Max2, 100), dtype="float32")
X_data2[:,1] = x_value2

X2_min = np.min(X_class[:,8])
X2_max = np.max(X_class[:,8])
X_value2 = np.array(np.linspace(X2_min, X2_max, 100), dtype="float32")



fig_g_x1 = plt.figure()
ax_g_x1 = fig_g_x1.add_subplot(1, 1, 1)
plt.xlim(X1_min,X1_max) 
ax_g_x1.set_xlabel("Systolic blood pressure",fontsize=8) 
ax_g_x1.set_ylabel(r"The value of $g$",fontsize=8)
ax_g_x1.tick_params(axis='both',labelsize=6)
# ----------------------------------
fig_g_x2 = plt.figure()
ax_g_x2 = fig_g_x2.add_subplot(1, 1, 1)
plt.xlim(X2_min,X2_max) 
ax_g_x2.set_xlabel("Diastolic blood pressure",fontsize=8) 
ax_g_x2.set_ylabel(r"The value of $g$",fontsize=8) 
ax_g_x2.tick_params(axis='both',labelsize=6)


# ---------------------
fig_g_diff = plt.figure()
ax_g_diff = fig_g_diff.add_subplot(1, 1, 1)
ax_g_diff.set_xlabel("subject",fontsize=8) 
ax_g_diff.set_ylabel(r"The difference of $g$",fontsize=8)
ax_g_diff.tick_params(axis='both',labelsize=6)

# -----------Deep-----------
fig_g_deep = plt.figure()
ax_g_deep = fig_g_deep.add_subplot(1, 1, 1)
ax_g_deep.set_title('(a)', fontsize=10) 
ax_g_deep.set_xlabel("subject",fontsize=8)  
ax_g_deep.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_deep.tick_params(axis='both',labelsize=6)


fig_g_deep_no = plt.figure()
ax_g_deep_no = fig_g_deep_no.add_subplot(1, 1, 1)
ax_g_deep_no.set_xlabel("subject",fontsize=8)   
ax_g_deep_no.set_ylabel(r"The estimate of $g$",fontsize=8)
ax_g_deep_no.tick_params(axis='both',labelsize=6)


# -----------Cox-----------
fig_g_Cox = plt.figure()
ax_g_Cox = fig_g_Cox.add_subplot(1, 1, 1)
ax_g_Cox.set_title('(b)', fontsize=10) 
ax_g_Cox.set_xlabel("subject",fontsize=8)  
ax_g_Cox.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_Cox.tick_params(axis='both',labelsize=6)


fig_g_Cox_no = plt.figure()
ax_g_Cox_no = fig_g_Cox_no.add_subplot(1, 1, 1)

ax_g_Cox_no.set_xlabel("subject",fontsize=8) 
ax_g_Cox_no.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_Cox_no.tick_params(axis='both',labelsize=6)


# -----------additive-----------
fig_g_additive = plt.figure()
ax_g_additive = fig_g_additive.add_subplot(1, 1, 1)
ax_g_additive.set_title('(c)', fontsize=10) 
ax_g_additive.set_xlabel("subject",fontsize=8) 
ax_g_additive.set_ylabel(r"The estimate of $g$",fontsize=8)
ax_g_additive.tick_params(axis='both',labelsize=6) 

fig_g_additive_no = plt.figure()
ax_g_additive_no = fig_g_additive_no.add_subplot(1, 1, 1)

ax_g_additive_no.set_xlabel("subject",fontsize=8)
ax_g_additive_no.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_additive_no.tick_params(axis='both',labelsize=6) 


# ------------training----------------
# ------------deep--------------
Est_Deep1 = Est_deep_judge(Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,X_data1,X_data2,Beta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0)

# Beta_deep
beta_D1 = Est_Deep1['Beta']
deep_Lambda_U = I_S(m,Est_Deep1['c'],U_R_train,nodevec)
deep_Lambda_V = I_S(m,Est_Deep1['c'],V_R_train,nodevec)
deep_a_b1 = LFD(Z_R_train[:,0],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,deep_Lambda_U,deep_Lambda_V,Est_Deep1['g_train'],beta_D1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
deep_a_b2 = LFD(Z_R_train[:,1],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,deep_Lambda_U,deep_Lambda_V,Est_Deep1['g_train'],beta_D1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
deep_a_b3 = LFD(Z_R_train[:,2],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,deep_Lambda_U,deep_Lambda_V,Est_Deep1['g_train'],beta_D1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
deep_a_b4 = LFD(Z_R_train[:,3],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,deep_Lambda_U,deep_Lambda_V,Est_Deep1['g_train'],beta_D1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
deep_a_b5 = LFD(Z_R_train[:,4],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,deep_Lambda_U,deep_Lambda_V,Est_Deep1['g_train'],beta_D1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
# I(beta_0)
deep_f_U = deep_Lambda_U * np.exp(np.dot(Z_R_train, beta_D1) + Est_Deep1['g_train'])
deep_f_V = deep_Lambda_V * np.exp(np.dot(Z_R_train, beta_D1) + Est_Deep1['g_train'])
deep_Ef_U = np.exp(- deep_f_U)
deep_Ef_V = np.exp(- deep_f_V)
deep_Q1_y = deep_Ef_U/(1-deep_Ef_U+1e-5)
deep_Q2_y = deep_Ef_U/(deep_Ef_U-deep_Ef_V+1e-5)
deep_Q3_y = deep_Ef_V/(deep_Ef_U-deep_Ef_V+1e-5)
deep_I_1 = deep_f_U*(De1_R_train*deep_Q1_y-De2_R_train*deep_Q2_y)*(Z_R_train[:,0]-deep_a_b1[:,0]) + deep_f_V*(De2_R_train*deep_Q3_y-De3_R_train)*(Z_R_train[:,0]-deep_a_b1[:,1])
deep_I_2 = deep_f_U*(De1_R_train*deep_Q1_y-De2_R_train*deep_Q2_y)*(Z_R_train[:,1]-deep_a_b2[:,0]) + deep_f_V*(De2_R_train*deep_Q3_y-De3_R_train)*(Z_R_train[:,1]-deep_a_b2[:,1])
deep_I_3 = deep_f_U*(De1_R_train*deep_Q1_y-De2_R_train*deep_Q2_y)*(Z_R_train[:,2]-deep_a_b3[:,0]) + deep_f_V*(De2_R_train*deep_Q3_y-De3_R_train)*(Z_R_train[:,2]-deep_a_b3[:,1])
deep_I_4 = deep_f_U*(De1_R_train*deep_Q1_y-De2_R_train*deep_Q2_y)*(Z_R_train[:,3]-deep_a_b4[:,0]) + deep_f_V*(De2_R_train*deep_Q3_y-De3_R_train)*(Z_R_train[:,3]-deep_a_b4[:,1])
deep_I_5 = deep_f_U*(De1_R_train*deep_Q1_y-De2_R_train*deep_Q2_y)*(Z_R_train[:,4]-deep_a_b5[:,0]) + deep_f_V*(De2_R_train*deep_Q3_y-De3_R_train)*(Z_R_train[:,4]-deep_a_b5[:,1])

deep_Info = np.zeros((5,5))
deep_Info[0,0] = np.mean(deep_I_1**2)
deep_Info[1,1] = np.mean(deep_I_2**2)
deep_Info[2,2] = np.mean(deep_I_3**2)
deep_Info[3,3] = np.mean(deep_I_4**2)
deep_Info[4,4] = np.mean(deep_I_5**2)
deep_Info[0,1] = np.mean(deep_I_1*deep_I_2)
deep_Info[1,0] = deep_Info[0,1]
deep_Info[0,2] = np.mean(deep_I_1*deep_I_3)
deep_Info[2,0] = deep_Info[0,2]
deep_Info[0,3] = np.mean(deep_I_1*deep_I_4)
deep_Info[3,0] = deep_Info[0,3]
deep_Info[0,4] = np.mean(deep_I_1*deep_I_5)
deep_Info[4,0] = deep_Info[0,4]
deep_Info[1,2] = np.mean(deep_I_2*deep_I_3)
deep_Info[2,1] = deep_Info[1,2]
deep_Info[1,3] = np.mean(deep_I_2*deep_I_4)
deep_Info[3,1] = deep_Info[1,3]
deep_Info[1,4] = np.mean(deep_I_2*deep_I_5)
deep_Info[4,1] = deep_Info[1,4]
deep_Info[2,3] = np.mean(deep_I_3*deep_I_4)
deep_Info[3,2] = deep_Info[2,3]
deep_Info[2,4] = np.mean(deep_I_3*deep_I_5)
deep_Info[4,2] = deep_Info[2,4]
deep_Info[3,4] = np.mean(deep_I_4*deep_I_5)
deep_Info[4,3] = deep_Info[3,4]
deep_Sigma = np.linalg.inv(deep_Info)/len(U_R_train)

dic_D = {"beta_deep": beta_D1, "sd_deep": np.sqrt(np.diag(deep_Sigma))}
Result_deep = pd.DataFrame(dic_D,index=['beta1','beta2','beta3','beta4','beta_age'])
Result_deep.to_csv('Result_deep.csv')
# -----------g_x----------------
ax_g_x1.plot(X_value1, Est_Deep1['g_value1'])
ax_g_x2.plot(X_value2, Est_Deep1['g_value2'])

fig_g_x1.savefig('fig_g_x1.jpeg', dpi=300, bbox_inches='tight')
fig_g_x2.savefig('fig_g_x2.jpeg', dpi=300, bbox_inches='tight')

# ------------Cox--------------
Z_new = np.array(df[["BMI01","GLUCOS01","HDL01","TCHSIU01","V1AGE01","SBPA21","SBPA22"]], dtype='float32')[A][np.arange(11000)]
X_new = np.array(df[["RACEGRP","GENDER","Cen1","Cen2","Cen3"]], dtype='float32')[A][np.arange(11000)]
Beta0_new = np.array(0.5*np.ones(Z_new.shape[1]), dtype='float32')
Est_L1 = Est_linear_judge(Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Beta0_new,nodevec,m,c0)
beta_L1 = Est_L1['Beta']
g_other5 = Est_L1['g_parameter']
L_Lambda_U = I_S(m,Est_L1['c'],U_R_train,nodevec)
L_Lambda_V = I_S(m,Est_L1['c'],V_R_train,nodevec)
L_a_b1 = LFD(Z_new[:,0],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
L_a_b2 = LFD(Z_new[:,1],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
L_a_b3 = LFD(Z_new[:,2],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
L_a_b4 = LFD(Z_new[:,3],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
L_a_b5 = LFD(Z_new[:,4],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
L_a_b6 = LFD(Z_new[:,5],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
L_a_b7 = LFD(Z_new[:,6],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,L_Lambda_U,L_Lambda_V,Est_L1['g_train'],beta_L1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
# I(beta_0)
L_f_U = L_Lambda_U * np.exp(np.dot(Z_new, beta_L1) + Est_L1['g_train'])
L_f_V = L_Lambda_V * np.exp(np.dot(Z_new, beta_L1) + Est_L1['g_train'])
L_Ef_U = np.exp(- L_f_U)
L_Ef_V = np.exp(- L_f_V)
L_Q1_y = L_Ef_U/(1-L_Ef_U+1e-5)
L_Q2_y = L_Ef_U/(L_Ef_U-L_Ef_V+1e-5)
L_Q3_y = L_Ef_V/(L_Ef_U-L_Ef_V+1e-5)
L_I_1 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,0]-L_a_b1[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,0]-L_a_b1[:,1])
L_I_2 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,1]-L_a_b2[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,1]-L_a_b2[:,1])
L_I_3 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,2]-L_a_b3[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,2]-L_a_b3[:,1])
L_I_4 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,3]-L_a_b4[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,3]-L_a_b4[:,1])
L_I_5 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,4]-L_a_b2[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,4]-L_a_b2[:,1])
L_I_6 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,5]-L_a_b3[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,5]-L_a_b3[:,1])
L_I_7 = L_f_U*(De1_R_train*L_Q1_y-De2_R_train*L_Q2_y)*(Z_new[:,6]-L_a_b4[:,0]) + L_f_V*(De2_R_train*L_Q3_y-De3_R_train)*(Z_new[:,6]-L_a_b4[:,1])

L_Info = np.zeros((7,7))
L_Info[0,0] = np.mean(L_I_1**2)
L_Info[1,1] = np.mean(L_I_2**2)
L_Info[2,2] = np.mean(L_I_3**2)
L_Info[3,3] = np.mean(L_I_4**2)
L_Info[4,4] = np.mean(L_I_5**2)
L_Info[5,5] = np.mean(L_I_6**2)
L_Info[6,6] = np.mean(L_I_7**2)
L_Info[0,1] = np.mean(L_I_1*L_I_2)
L_Info[1,0] = L_Info[0,1]
L_Info[0,2] = np.mean(L_I_1*L_I_3)
L_Info[2,0] = L_Info[0,2]
L_Info[0,3] = np.mean(L_I_1*L_I_4)
L_Info[3,0] = L_Info[0,3]
L_Info[0,4] = np.mean(L_I_1*L_I_5)
L_Info[4,0] = L_Info[0,4]
L_Info[0,5] = np.mean(L_I_1*L_I_6)
L_Info[5,0] = L_Info[0,5]
L_Info[0,6] = np.mean(L_I_1*L_I_7)
L_Info[6,0] = L_Info[0,6]
L_Info[1,2] = np.mean(L_I_2*L_I_3)
L_Info[2,1] = L_Info[1,2]
L_Info[1,3] = np.mean(L_I_2*L_I_4)
L_Info[3,1] = L_Info[1,3]
L_Info[1,4] = np.mean(L_I_2*L_I_5)
L_Info[4,1] = L_Info[1,4]
L_Info[1,5] = np.mean(L_I_2*L_I_6)
L_Info[5,1] = L_Info[1,5]
L_Info[1,6] = np.mean(L_I_2*L_I_7)
L_Info[6,1] = L_Info[1,6]
L_Info[2,3] = np.mean(L_I_3*L_I_4)
L_Info[3,2] = L_Info[2,3]
L_Info[2,4] = np.mean(L_I_3*L_I_5)
L_Info[4,2] = L_Info[2,4]
L_Info[2,5] = np.mean(L_I_3*L_I_6)
L_Info[5,2] = L_Info[2,5]
L_Info[2,6] = np.mean(L_I_3*L_I_7)
L_Info[6,2] = L_Info[2,6]
L_Info[3,4] = np.mean(L_I_4*L_I_5)
L_Info[4,3] = L_Info[3,4]
L_Info[3,5] = np.mean(L_I_4*L_I_6)
L_Info[5,3] = L_Info[3,5]
L_Info[3,6] = np.mean(L_I_4*L_I_7)
L_Info[6,3] = L_Info[3,6]
L_Info[4,5] = np.mean(L_I_5*L_I_6)
L_Info[5,4] = L_Info[4,5]
L_Info[4,6] = np.mean(L_I_5*L_I_7)
L_Info[6,4] = L_Info[4,6]
L_Info[5,6] = np.mean(L_I_6*L_I_7)
L_Info[6,5] = L_Info[5,6]

L_Sigma = np.linalg.inv(L_Info)/len(U_R_train)

dic_L = {"beta_L": beta_L1, "sd_L": np.sqrt(np.diag(L_Sigma))}
Result_L = pd.DataFrame(dic_L,index=['beta1','beta2','beta3','beta4','beta_age','beta_systolic','beta_diastolic'])
Result_L.to_csv('Result_L.csv')


dic_g5 = {"g_5": g_other5}
Result_g5 = pd.DataFrame(dic_g5,index=['g1','g2','g3','g4','g5'])
Result_g5.to_csv('Result_g5.csv')


# ------------additive-------------
Est_additive1 = Est_additive_judge(Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Beta0,nodevec,m,c0,m0,nodevec0)

# Beta_additive
beta_A1 = Est_additive1['Beta']
additive_Lambda_U = I_S(m,Est_additive1['c'],U_R_train,nodevec)
additive_Lambda_V = I_S(m,Est_additive1['c'],V_R_train,nodevec)
additive_a_b1 = LFD(Z_R_train[:,0],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,additive_Lambda_U,additive_Lambda_V,Est_additive1['g_train'],beta_A1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
additive_a_b2 = LFD(Z_R_train[:,1],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,additive_Lambda_U,additive_Lambda_V,Est_additive1['g_train'],beta_A1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
additive_a_b3 = LFD(Z_R_train[:,2],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,additive_Lambda_U,additive_Lambda_V,Est_additive1['g_train'],beta_A1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
additive_a_b4 = LFD(Z_R_train[:,3],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,additive_Lambda_U,additive_Lambda_V,Est_additive1['g_train'],beta_A1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
additive_a_b5 = LFD(Z_R_train[:,4],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,additive_Lambda_U,additive_Lambda_V,Est_additive1['g_train'],beta_A1,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
# I(beta_0)
additive_f_U = additive_Lambda_U * np.exp(np.dot(Z_R_train, beta_A1) + Est_additive1['g_train'])
additive_f_V = additive_Lambda_V * np.exp(np.dot(Z_R_train, beta_A1) + Est_additive1['g_train'])
additive_Ef_U = np.exp(- additive_f_U)
additive_Ef_V = np.exp(- additive_f_V)
additive_Q1_y = additive_Ef_U/(1-additive_Ef_U+1e-5)
additive_Q2_y = additive_Ef_U/(additive_Ef_U-additive_Ef_V+1e-5)
additive_Q3_y = additive_Ef_V/(additive_Ef_U-additive_Ef_V+1e-5)
additive_I_1 = additive_f_U*(De1_R_train*additive_Q1_y-De2_R_train*additive_Q2_y)*(Z_R_train[:,0]-additive_a_b1[:,0]) + additive_f_V*(De2_R_train*additive_Q3_y-De3_R_train)*(Z_R_train[:,0]-additive_a_b1[:,1])
additive_I_2 = additive_f_U*(De1_R_train*additive_Q1_y-De2_R_train*additive_Q2_y)*(Z_R_train[:,1]-additive_a_b2[:,0]) + additive_f_V*(De2_R_train*additive_Q3_y-De3_R_train)*(Z_R_train[:,1]-additive_a_b2[:,1])
additive_I_3 = additive_f_U*(De1_R_train*additive_Q1_y-De2_R_train*additive_Q2_y)*(Z_R_train[:,2]-additive_a_b3[:,0]) + additive_f_V*(De2_R_train*additive_Q3_y-De3_R_train)*(Z_R_train[:,2]-additive_a_b3[:,1])
additive_I_4 = additive_f_U*(De1_R_train*additive_Q1_y-De2_R_train*additive_Q2_y)*(Z_R_train[:,3]-additive_a_b4[:,0]) + additive_f_V*(De2_R_train*additive_Q3_y-De3_R_train)*(Z_R_train[:,3]-additive_a_b4[:,1])
additive_I_5 = additive_f_U*(De1_R_train*additive_Q1_y-De2_R_train*additive_Q2_y)*(Z_R_train[:,4]-additive_a_b5[:,0]) + additive_f_V*(De2_R_train*additive_Q3_y-De3_R_train)*(Z_R_train[:,4]-additive_a_b5[:,1])

additive_Info = np.zeros((5,5))
additive_Info[0,0] = np.mean(additive_I_1**2)
additive_Info[1,1] = np.mean(additive_I_2**2)
additive_Info[2,2] = np.mean(additive_I_3**2)
additive_Info[3,3] = np.mean(additive_I_4**2)
additive_Info[4,4] = np.mean(additive_I_5**2)
additive_Info[0,1] = np.mean(additive_I_1*additive_I_2)
additive_Info[1,0] = additive_Info[0,1]
additive_Info[0,2] = np.mean(additive_I_1*additive_I_3)
additive_Info[2,0] = additive_Info[0,2]
additive_Info[0,3] = np.mean(additive_I_1*additive_I_4)
additive_Info[3,0] = additive_Info[0,3]
additive_Info[0,4] = np.mean(additive_I_1*additive_I_5)
additive_Info[4,0] = additive_Info[0,4]
additive_Info[1,2] = np.mean(additive_I_2*additive_I_3)
additive_Info[2,1] = additive_Info[1,2]
additive_Info[1,3] = np.mean(additive_I_2*additive_I_4)
additive_Info[3,1] = additive_Info[1,3]
additive_Info[1,4] = np.mean(additive_I_2*additive_I_5)
additive_Info[4,1] = additive_Info[1,4]
additive_Info[2,3] = np.mean(additive_I_3*additive_I_4)
additive_Info[3,2] = additive_Info[2,3]
additive_Info[2,4] = np.mean(additive_I_3*additive_I_5)
additive_Info[4,2] = additive_Info[2,4]
additive_Info[3,4] = np.mean(additive_I_4*additive_I_5)
additive_Info[4,3] = additive_Info[3,4]
additive_Sigma = np.linalg.inv(additive_Info)/len(U_R_train)

dic_additive = {"beta_additive": beta_A1, "sd_additive": np.sqrt(np.diag(additive_Sigma))}
Result_additive = pd.DataFrame(dic_additive,index=['beta1','beta2','beta3','beta4','beta_age'])
Result_additive.to_csv('Result_additive.csv')


# -----------------g--------------------
ax_g_diff.scatter(np.arange(X_R_train.shape[0]), Est_Deep1['g_train']-Est_L1['g_train'], s=4)
fig_g_diff.savefig('fig_g_diff.jpeg', dpi=300, bbox_inches='tight')
# -----------------------------------------------
ax_g_deep.scatter(np.arange(X_R_train.shape[0]), Est_Deep1['g_train'], s=4)
fig_g_deep.savefig('fig_g_deep.jpeg', dpi=300, bbox_inches='tight')

ax_g_Cox.scatter(np.arange(X_R_train.shape[0]), Est_L1['g_train'], s=4)
fig_g_Cox.savefig('fig_g_Cox.jpeg', dpi=300, bbox_inches='tight')

ax_g_additive.scatter(np.arange(X_R_train.shape[0]), Est_L1['g_train'], s=4)
fig_g_additive.savefig('fig_g_additive.jpeg', dpi=300, bbox_inches='tight')

ax_g_deep_no.scatter(np.arange(X_R_train.shape[0]), Est_Deep1['g_train'], s=4)
fig_g_deep_no.savefig('fig_g_deep_no.jpeg', dpi=300, bbox_inches='tight')

ax_g_Cox_no.scatter(np.arange(X_R_train.shape[0]), Est_L1['g_train'], s=4)
fig_g_Cox_no.savefig('fig_g_Cox_no.jpeg', dpi=300, bbox_inches='tight')

ax_g_additive_no.scatter(np.arange(X_R_train.shape[0]), Est_L1['g_train'], s=4)
fig_g_additive_no.savefig('fig_g_additive_no.jpeg', dpi=300, bbox_inches='tight')

#%% ==================================

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_title(r'$(a)\quad \hat\theta_1$', fontsize=10) 
ax1.set_xlabel("Fold",fontsize=8) 
ax1.set_ylabel("Estimates of effect",fontsize=8) 
ax1.tick_params(axis='both',labelsize=6) 
ax1.grid(True)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title(r'$(b)\quad \hat\theta_2$', fontsize=10) 
ax2.set_xlabel("Fold",fontsize=8) 
ax2.set_ylabel("Estimates of effect",fontsize=8) 
ax2.tick_params(axis='both',labelsize=6) 
ax2.grid(True)

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_title(r'$(c)\quad \hat\theta_3$', fontsize=10)
ax3.set_xlabel("Fold",fontsize=8)
ax3.set_ylabel("Estimates of effect",fontsize=8)
ax3.tick_params(axis='both',labelsize=6)
ax3.grid(True)

fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.set_title(r'$(d)\quad \hat\theta_4$', fontsize=10)
ax4.set_xlabel("Fold",fontsize=8)
ax4.set_ylabel("Estimates of effect",fontsize=8)
ax4.tick_params(axis='both',labelsize=6)
ax4.grid(True)


fig5 = plt.figure()
ax5 = fig5.add_subplot(1, 1, 1)
ax5.set_title(r'$(e)\quad \hat\theta_5$', fontsize=10)
ax5.set_xlabel("Fold",fontsize=8)
ax5.set_ylabel("Estimates of effect",fontsize=8)
ax5.tick_params(axis='both',labelsize=6)
ax5.grid(True)


Beta_g_D = np.zeros((5,len(U_R_test)))
Beta_g_L = np.zeros((5,len(U_R_test)))
Beta_g_A = np.zeros((5,len(U_R_test)))

Beta_g_D1 = np.zeros((5,len(V1)))
Beta_g_L1 = np.zeros((5,len(V1)))
Beta_g_A1 = np.zeros((5,len(V1)))

Beta_g_D0 = np.zeros((5,len(V1)))
Beta_g_L0 = np.zeros((5,len(V1)))
Beta_g_A0 = np.zeros((5,len(V1)))

C_D = np.zeros((5, m+3))
C_L = np.zeros((5, m+3))
C_A = np.zeros((5, m+3))

c_n= 2200
for i in range(5):
    print('i =', i)
    Z_train = np.delete(Z_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    X_train = np.delete(X_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    U_train = np.delete(U_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    V_train = np.delete(V_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De1_train = np.delete(De1_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De2_train = np.delete(De2_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De3_train = np.delete(De3_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    n = len(U_train)
    
    #%% DPLCM
    Est_D = Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_R_test,X_sort1_subject,X_sort0_subject,Beta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0)
    
    Beta_deep = Est_D['Beta']
    Lambda_U = I_S(m,Est_D['c'],U_train,nodevec)
    Lambda_V = I_S(m,Est_D['c'],V_train,nodevec)
    a_b1 = LFD(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U,Lambda_V,Est_D['g_train'],Beta_deep,n_layer,n_node=100,n_lr=1e-4,n_epoch=200)
    a_b2 = LFD(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U,Lambda_V,Est_D['g_train'],Beta_deep,n_layer,n_node=100,n_lr=1e-4,n_epoch=200)
    a_b3 = LFD(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U,Lambda_V,Est_D['g_train'],Beta_deep,n_layer,n_node=100,n_lr=1e-4,n_epoch=200)
    a_b4 = LFD(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U,Lambda_V,Est_D['g_train'],Beta_deep,n_layer,n_node=100,n_lr=5e-5,n_epoch=200)
    a_b5 = LFD(Z_train[:,4],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U,Lambda_V,Est_D['g_train'],Beta_deep,n_layer,n_node=100,n_lr=3e-5,n_epoch=200)
    
    f_U = Lambda_U * np.exp(np.dot(Z_train, Beta_deep) + Est_D['g_train'])
    f_V = Lambda_V * np.exp(np.dot(Z_train, Beta_deep) + Est_D['g_train'])
    Ef_U = np.exp(- f_U)
    Ef_V = np.exp(- f_V)
    Q1_y = Ef_U/(1-Ef_U+1e-5)
    Q2_y = Ef_U/(Ef_U-Ef_V+1e-5)
    Q3_y = Ef_V/(Ef_U-Ef_V+1e-5)
    I_1 = f_U*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train[:,0]-a_b1[:,0]) + f_V*(De2_train*Q3_y-De3_train)*(Z_train[:,0]-a_b1[:,1])
    I_2 = f_U*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train[:,1]-a_b2[:,0]) + f_V*(De2_train*Q3_y-De3_train)*(Z_train[:,1]-a_b2[:,1])
    I_3 = f_U*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train[:,2]-a_b3[:,0]) + f_V*(De2_train*Q3_y-De3_train)*(Z_train[:,2]-a_b3[:,1])
    I_4 = f_U*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train[:,3]-a_b4[:,0]) + f_V*(De2_train*Q3_y-De3_train)*(Z_train[:,3]-a_b4[:,1])
    I_5 = f_U*(De1_train*Q1_y-De2_train*Q2_y)*(Z_train[:,4]-a_b5[:,0]) + f_V*(De2_train*Q3_y-De3_train)*(Z_train[:,4]-a_b5[:,1])
    
    Info = np.zeros((5,5))
    Info[0,0] = np.mean(I_1**2)
    Info[1,1] = np.mean(I_2**2)
    Info[2,2] = np.mean(I_3**2)
    Info[3,3] = np.mean(I_4**2)
    Info[4,4] = np.mean(I_5**2)
    Info[0,1] = np.mean(I_1*I_2)
    Info[1,0] = Info[0,1]
    Info[0,2] = np.mean(I_1*I_3)
    Info[2,0] = Info[0,2]
    Info[0,3] = np.mean(I_1*I_4)
    Info[3,0] = Info[0,3]
    Info[0,4] = np.mean(I_1*I_5)
    Info[4,0] = Info[0,4]
    Info[1,2] = np.mean(I_2*I_3)
    Info[2,1] = Info[1,2]
    Info[1,3] = np.mean(I_2*I_4)
    Info[3,1] = Info[1,3]
    Info[1,4] = np.mean(I_2*I_5)
    Info[4,1] = Info[1,4]
    Info[2,3] = np.mean(I_3*I_4)
    Info[3,2] = Info[2,3]
    Info[2,4] = np.mean(I_3*I_5)
    Info[4,2] = Info[2,4]
    Info[3,4] = np.mean(I_4*I_5)
    Info[4,3] = Info[3,4]
    Sigma = np.linalg.inv(Info)/n
    sd1 = np.sqrt(Sigma[0,0])
    sd2 = np.sqrt(Sigma[1,1])
    sd3 = np.sqrt(Sigma[2,2])
    sd4 = np.sqrt(Sigma[3,3])
    sd5 = np.sqrt(Sigma[4,4])
    # ----------------------
    y_min1 = Beta_deep[0] - 1.96*sd1
    y_max1 = Beta_deep[0] + 1.96*sd1
    ax1.plot(i+1-0.1, Beta_deep[0], marker='o', markersize=4, ls='-', color='blue', label='DNN-based')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    ax1.plot((i+1-0.1)*np.ones(2), np.array([y_min1, y_max1]), marker='_', ls='-', color='blue')
    # -----------------------
    y_min2 = Beta_deep[1] - 1.96*sd2
    y_max2 = Beta_deep[1] + 1.96*sd2
    ax2.plot(i+1-0.1, Beta_deep[1], marker='o', markersize=4, ls='-',color='blue', label='DNN-based')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    ax2.plot((i+1-0.1)*np.ones(2), np.array([y_min2, y_max2]), marker='_', ls='-', color='blue')
    # -----------------------
    y_min3 = Beta_deep[2] - 1.96*sd3
    y_max3 = Beta_deep[2] + 1.96*sd3
    ax3.plot(i+1-0.1, Beta_deep[2], marker='o', markersize=4, ls='-', color='blue', label='DNN-based')
    if (i == 0):
        ax3.legend(loc='best', fontsize=6)
    ax3.plot((i+1-0.1)*np.ones(2), np.array([y_min3, y_max3]), marker='_', ls='-', color='blue')
    # -----------------------
    y_min4 = Beta_deep[3] - 1.96*sd4
    y_max4 = Beta_deep[3] + 1.96*sd4
    ax4.plot(i+1-0.1, Beta_deep[3], marker='o', markersize=4, ls='-', color='blue', label='DNN-based')
    if (i == 0):
        ax4.legend(loc='best', fontsize=6)
    ax4.plot((i+1-0.1)*np.ones(2), np.array([y_min4, y_max4]), marker='_', ls='-', color='blue')
    # ----------------------
    y_min5 = Beta_deep[4] - 1.96*sd5
    y_max5 = Beta_deep[4] + 1.96*sd5
    ax5.plot(i+1-0.1, Beta_deep[4], marker='o', markersize=4, ls='-', color='blue', label='DNN-based')
    if (i == 0):
        ax5.legend(loc='best', fontsize=6)
    ax5.plot((i+1-0.1)*np.ones(2), np.array([y_min5, y_max5]), marker='_', ls='-', color='blue')
    
    Beta_g_D[i] = np.dot(Z_R_test, Beta_deep) + Est_D['g_test']
    Beta_g_D1[i] = np.dot(Z_sort1_subject, Beta_deep) + Est_D['g_test1']
    Beta_g_D0[i] = np.dot(Z_sort0_subject, Beta_deep) + Est_D['g_test0']
    C_D[i] = Est_D['c']
    
    #%% CPH
    Est_L = Est_linear(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_R_test,X_sort1_subject,X_sort0_subject,Beta0,nodevec,m,c0)
    
    Beta_L = Est_L['Beta']
    Lambda_U_L = I_S(m,Est_L['c'],U_train,nodevec)
    Lambda_V_L = I_S(m,Est_L['c'],V_train,nodevec)
    a_b1_L = LFD(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_L,Lambda_V_L,Est_L['g_train'],Beta_L,n_layer,n_node=100,n_lr=5e-4,n_epoch=500)
    a_b2_L = LFD(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_L,Lambda_V_L,Est_L['g_train'],Beta_L,n_layer,n_node=100,n_lr=5e-4,n_epoch=500)
    a_b3_L = LFD(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_L,Lambda_V_L,Est_L['g_train'],Beta_L,n_layer,n_node=100,n_lr=5e-4,n_epoch=500)
    a_b4_L = LFD(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_L,Lambda_V_L,Est_L['g_train'],Beta_L,n_layer,n_node=100,n_lr=5e-4,n_epoch=500)
    a_b5_L = LFD(Z_train[:,4],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_L,Lambda_V_L,Est_L['g_train'],Beta_L,n_layer,n_node=100,n_lr=5e-4,n_epoch=500)
    
    f_U_L = Lambda_U_L * np.exp(np.dot(Z_train, Beta_L) + Est_L['g_train'])
    f_V_L = Lambda_V_L * np.exp(np.dot(Z_train, Beta_L) + Est_L['g_train'])
    Ef_U_L = np.exp(- f_U_L)
    Ef_V_L = np.exp(- f_V_L)
    Q1_y_L = Ef_U_L/(1-Ef_U_L+1e-5)
    Q2_y_L = Ef_U_L/(Ef_U_L-Ef_V_L+1e-5)
    Q3_y_L = Ef_V_L/(Ef_U_L-Ef_V_L+1e-5)
    I_1_L = f_U_L*(De1_train*Q1_y_L-De2_train*Q2_y_L)*(Z_train[:,0]-a_b1_L[:,0]) + f_V_L*(De2_train*Q3_y_L-De3_train)*(Z_train[:,0]-a_b1_L[:,1])
    I_2_L = f_U_L*(De1_train*Q1_y_L-De2_train*Q2_y_L)*(Z_train[:,1]-a_b2_L[:,0]) + f_V_L*(De2_train*Q3_y_L-De3_train)*(Z_train[:,1]-a_b2_L[:,1])
    I_3_L = f_U_L*(De1_train*Q1_y_L-De2_train*Q2_y_L)*(Z_train[:,2]-a_b3_L[:,0]) + f_V_L*(De2_train*Q3_y_L-De3_train)*(Z_train[:,2]-a_b3_L[:,1])
    I_4_L = f_U_L*(De1_train*Q1_y_L-De2_train*Q2_y_L)*(Z_train[:,3]-a_b4_L[:,0]) + f_V_L*(De2_train*Q3_y_L-De3_train)*(Z_train[:,3]-a_b4_L[:,1])
    I_5_L = f_U_L*(De1_train*Q1_y_L-De2_train*Q2_y_L)*(Z_train[:,4]-a_b5_L[:,0]) + f_V_L*(De2_train*Q3_y_L-De3_train)*(Z_train[:,4]-a_b5_L[:,1])
    
    Info_L = np.zeros((5,5))
    Info_L[0,0] = np.mean(I_1_L**2)
    Info_L[1,1] = np.mean(I_2_L**2)
    Info_L[2,2] = np.mean(I_3_L**2)
    Info_L[3,3] = np.mean(I_4_L**2)
    Info_L[4,4] = np.mean(I_5_L**2)
    Info_L[0,1] = np.mean(I_1_L*I_2_L)
    Info_L[1,0] = Info_L[0,1]
    Info_L[0,2] = np.mean(I_1_L*I_3_L)
    Info_L[2,0] = Info_L[0,2]
    Info_L[0,3] = np.mean(I_1_L*I_4_L)
    Info_L[3,0] = Info_L[0,3]
    Info_L[0,4] = np.mean(I_1_L*I_5_L)
    Info_L[4,0] = Info_L[0,4]
    Info_L[1,2] = np.mean(I_2_L*I_3_L)
    Info_L[2,1] = Info_L[1,2]
    Info_L[1,3] = np.mean(I_2_L*I_4_L)
    Info_L[3,1] = Info_L[1,3]
    Info_L[1,4] = np.mean(I_2_L*I_5_L)
    Info_L[4,1] = Info_L[1,4]
    Info_L[2,3] = np.mean(I_3_L*I_4_L)
    Info_L[3,2] = Info_L[2,3]
    Info_L[2,4] = np.mean(I_3_L*I_5_L)
    Info_L[4,2] = Info_L[2,4]
    Info_L[3,4] = np.mean(I_4_L*I_5_L)
    Info_L[4,3] = Info_L[3,4]
    Sigma_L = np.linalg.inv(Info_L)/n
    sd1_L = np.sqrt(Sigma_L[0,0])
    sd2_L = np.sqrt(Sigma_L[1,1])
    sd3_L = np.sqrt(Sigma_L[2,2])
    sd4_L = np.sqrt(Sigma_L[3,3])
    sd5_L = np.sqrt(Sigma_L[4,4])
    # ----------------------
    y_min1 = Beta_L[0] - 1.96*sd1_L
    y_max1 = Beta_L[0] + 1.96*sd1_L
    ax1.plot(i+1, Beta_L[0], marker='s', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    ax1.plot((i+1)*np.ones(2), np.array([y_min1, y_max1]), marker='_', ls='-', color='orange')
    # -----------------------
    y_min2 = Beta_L[1] - 1.96*sd2_L
    y_max2 = Beta_L[1] + 1.96*sd2_L
    ax2.plot(i+1, Beta_L[1], marker='s', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    ax2.plot((i+1)*np.ones(2), np.array([y_min2, y_max2]), marker='_', ls='-', color='orange')
    # -----------------------
    y_min3 = Beta_L[2] - 1.96*sd3_L
    y_max3 = Beta_L[2] + 1.96*sd3_L
    ax3.plot(i+1, Beta_L[2], marker='s', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax3.legend(loc='best', fontsize=6)
    ax3.plot((i+1)*np.ones(2), np.array([y_min3, y_max3]), marker='_', ls='-', color='orange')
    # -----------------------
    y_min4 = Beta_L[3] - 1.96*sd4_L
    y_max4 = Beta_L[3] + 1.96*sd4_L
    ax4.plot(i+1, Beta_L[3], marker='s', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax4.legend(loc='best', fontsize=6)
    ax4.plot((i+1)*np.ones(2), np.array([y_min4, y_max4]), marker='_', ls='-', color='orange')
    # -----------------------
    y_min5 = Beta_L[4] - 1.96*sd5_L
    y_max5 = Beta_L[4] + 1.96*sd5_L
    ax5.plot(i+1, Beta_L[4], marker='s', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax5.legend(loc='best', fontsize=6)
    ax5.plot((i+1)*np.ones(2), np.array([y_min5, y_max5]), marker='_', ls='-', color='orange')
    
    Beta_g_L[i] = np.dot(Z_R_test, Beta_L) + Est_L['g_test']
    Beta_g_L1[i] = np.dot(Z_sort1_subject, Beta_L) + Est_L['g_test1']
    Beta_g_L0[i] = np.dot(Z_sort0_subject, Beta_L) + Est_L['g_test0']
    C_L[i] = Est_L['c']


    #%% PLACM
    Est_A = Est_additive(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_R_test,X_sort1_subject,X_sort0_subject,Beta0,nodevec,m,c0,m0,nodevec0)
    
    Beta_A = Est_A['Beta']
    Lambda_U_A = I_S(m,Est_A['c'],U_train,nodevec)
    Lambda_V_A = I_S(m,Est_A['c'],V_train,nodevec)
    a_b1_A = LFD(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_A,Lambda_V_A,Est_A['g_train'],Beta_A,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
    a_b2_A = LFD(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_A,Lambda_V_A,Est_A['g_train'],Beta_A,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
    a_b3_A = LFD(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_A,Lambda_V_A,Est_A['g_train'],Beta_A,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
    a_b4_A = LFD(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_A,Lambda_V_A,Est_A['g_train'],Beta_A,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
    a_b5_A = LFD(Z_train[:,4],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Lambda_U_A,Lambda_V_A,Est_A['g_train'],Beta_A,n_layer,n_node=100,n_lr=1e-3,n_epoch=500)
    
    f_U_A = Lambda_U_A * np.exp(np.dot(Z_train, Beta_A) + Est_A['g_train'])
    f_V_A = Lambda_V_A * np.exp(np.dot(Z_train, Beta_A) + Est_A['g_train'])
    Ef_U_A = np.exp(- f_U_A)
    Ef_V_A = np.exp(- f_V_A)
    Q1_y_A = Ef_U_A/(1-Ef_U_A+1e-5)
    Q2_y_A = Ef_U_A/(Ef_U_A-Ef_V_A+1e-5)
    Q3_y_A = Ef_V_A/(Ef_U_A-Ef_V_A+1e-5)
    I_1_A = f_U_A*(De1_train*Q1_y_A-De2_train*Q2_y_A)*(Z_train[:,0]-a_b1_A[:,0]) + f_V_A*(De2_train*Q3_y_A-De3_train)*(Z_train[:,0]-a_b1_A[:,1])
    I_2_A = f_U_A*(De1_train*Q1_y_A-De2_train*Q2_y_A)*(Z_train[:,1]-a_b2_A[:,0]) + f_V_A*(De2_train*Q3_y_A-De3_train)*(Z_train[:,1]-a_b2_A[:,1])
    I_3_A = f_U_A*(De1_train*Q1_y_A-De2_train*Q2_y_A)*(Z_train[:,2]-a_b3_A[:,0]) + f_V_A*(De2_train*Q3_y_A-De3_train)*(Z_train[:,2]-a_b3_A[:,1])
    I_4_A = f_U_A*(De1_train*Q1_y_A-De2_train*Q2_y_A)*(Z_train[:,3]-a_b4_A[:,0]) + f_V_A*(De2_train*Q3_y_A-De3_train)*(Z_train[:,3]-a_b4_A[:,1])
    I_5_A = f_U_A*(De1_train*Q1_y_A-De2_train*Q2_y_A)*(Z_train[:,4]-a_b5_A[:,0]) + f_V_A*(De2_train*Q3_y_A-De3_train)*(Z_train[:,4]-a_b5_A[:,1])
    
    Info_A = np.zeros((5,5))
    Info_A[0,0] = np.mean(I_1_A**2)
    Info_A[1,1] = np.mean(I_2_A**2)
    Info_A[2,2] = np.mean(I_3_A**2)
    Info_A[3,3] = np.mean(I_4_A**2)
    Info_A[4,4] = np.mean(I_5_A**2)
    Info_A[0,1] = np.mean(I_1_A*I_2_A)
    Info_A[1,0] = Info_A[0,1]
    Info_A[0,2] = np.mean(I_1_A*I_3_A)
    Info_A[2,0] = Info_A[0,2]
    Info_A[0,3] = np.mean(I_1_A*I_4_A)
    Info_A[3,0] = Info_A[0,3]
    Info_A[0,4] = np.mean(I_1_A*I_5_A)
    Info_A[4,0] = Info_A[0,4]
    Info_A[1,2] = np.mean(I_2_A*I_3_A)
    Info_A[2,1] = Info_A[1,2]
    Info_A[1,3] = np.mean(I_2_A*I_4_A)
    Info_A[3,1] = Info_A[1,3]
    Info_A[1,4] = np.mean(I_2_A*I_5_A)
    Info_A[4,1] = Info_A[1,4]
    Info_A[2,3] = np.mean(I_3_A*I_4_A)
    Info_A[3,2] = Info_A[2,3]
    Info_A[2,4] = np.mean(I_3_A*I_5_A)
    Info_A[4,2] = Info_A[2,4]
    Info_A[3,4] = np.mean(I_4_A*I_5_A)
    Info_A[4,3] = Info_A[3,4]
    Sigma_A = np.linalg.inv(Info_A)/n
    sd1_A = np.sqrt(Sigma_A[0,0])
    sd2_A = np.sqrt(Sigma_A[1,1])
    sd3_A = np.sqrt(Sigma_A[2,2])
    sd4_A = np.sqrt(Sigma_A[3,3])
    sd5_A = np.sqrt(Sigma_A[4,4])
    # ----------------------
    y_min1 = Beta_A[0] - 1.96*sd1_A
    y_max1 = Beta_A[0] + 1.96*sd1_A
    ax1.plot(i+1+0.1, Beta_A[0], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    ax1.plot((i+1+0.1)*np.ones(2), np.array([y_min1, y_max1]), marker='_', ls='-', color='green')
    # -----------------------
    y_min2 = Beta_A[1] - 1.96*sd2_A
    y_max2 = Beta_A[1] + 1.96*sd2_A
    ax2.plot(i+1+0.1, Beta_A[1], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    ax2.plot((i+1+0.1)*np.ones(2), np.array([y_min2, y_max2]), marker='_', ls='-', color='green')
    # -----------------------
    y_min3 = Beta_A[2] - 1.96*sd3_A
    y_max3 = Beta_A[2] + 1.96*sd3_A
    ax3.plot(i+1+0.1, Beta_A[2], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax3.legend(loc='best', fontsize=6)
    ax3.plot((i+1+0.1)*np.ones(2), np.array([y_min3, y_max3]), marker='_', ls='-', color='green')
    # -----------------------
    y_min4 = Beta_A[3] - 1.96*sd4_A
    y_max4 = Beta_A[3] + 1.96*sd4_A
    ax4.plot(i+1+0.1, Beta_A[3], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax4.legend(loc='best', fontsize=6)
    ax4.plot((i+1+0.1)*np.ones(2), np.array([y_min4, y_max4]), marker='_', ls='-', color='green')
    # -----------------------
    y_min5 = Beta_A[4] - 1.96*sd5_A
    y_max5 = Beta_A[4] + 1.96*sd5_A
    ax5.plot(i+1+0.1, Beta_A[4], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax5.legend(loc='best', fontsize=6)
    ax5.plot((i+1+0.1)*np.ones(2), np.array([y_min5, y_max5]), marker='_', ls='-', color='green')
    
    Beta_g_A[i] = np.dot(Z_R_test, Beta_A) + Est_A['g_test']
    Beta_g_A1[i] = np.dot(Z_sort1_subject, Beta_A) + Est_A['g_test1']
    Beta_g_A0[i] = np.dot(Z_sort0_subject, Beta_A) + Est_A['g_test0']
    C_A[i] = Est_A['c']

# ===============================================
fig1.savefig('fig1.jpeg', dpi=300, bbox_inches='tight')
fig2.savefig('fig2.jpeg', dpi=300, bbox_inches='tight')
fig3.savefig('fig3.jpeg', dpi=300, bbox_inches='tight')
fig4.savefig('fig4.jpeg', dpi=300, bbox_inches='tight')
fig5.savefig('fig5.jpeg', dpi=300, bbox_inches='tight')
# =======================================================

#----------Deep-----------
C_D = np.mean(C_D,axis=0)
B_g_deep = np.mean(Beta_g_D,axis=0)
Lamd_U_deep = I_S(m,C_D,U_R_test,nodevec)
Lamd_V_deep = I_S(m,C_D,V_R_test,nodevec)

#----------Linear-----------
C_L = np.mean(C_L,axis=0)
B_g_L = np.mean(Beta_g_L,axis=0)
Lamd_U_L = I_S(m,C_L,U_R_test,nodevec)
Lamd_V_L = I_S(m,C_L,V_R_test,nodevec)

#----------Additive-----------
C_A = np.mean(C_A,axis=0)
B_g_A = np.mean(Beta_g_A,axis=0) 
Lamd_U_A = I_S(m,C_A,U_R_test,nodevec) 
Lamd_V_A = I_S(m,C_A,V_R_test,nodevec) 


S_U_deep = np.exp(- Lamd_U_deep * np.exp(B_g_deep))
S_V_deep = np.exp(- Lamd_V_deep * np.exp(B_g_deep))
S_U_L = np.exp(- Lamd_U_L * np.exp(B_g_L))
S_V_L = np.exp(- Lamd_V_L * np.exp(B_g_L))
S_U_A = np.exp(- Lamd_U_A * np.exp(B_g_A))
S_V_A = np.exp(- Lamd_V_A * np.exp(B_g_A))

N1_deep = 0
N2_deep = 0
N3_deep = 0
N1_L = 0
N2_L = 0
N3_L = 0
N1_A = 0
N2_A = 0
N3_A = 0
for k in range(len(U_R_test)):
    if (De1_R_test[k] == 1):
        if (1 - S_V_deep[k] > 0.5):
            N1_deep = N1_deep + 1
        if (1 - S_V_L[k] > 0.5):
            N1_L = N1_L + 1
        if (1 - S_V_A[k] > 0.5):
            N1_A = N1_A + 1
    elif (De2_R_test[k] == 1):
        if (S_V_deep[k] < 0.5):
            N2_deep = N2_deep + 1
        if (S_V_L[k] < 0.5):
            N2_L = N2_L + 1
        if (S_V_A[k] < 0.5):
            N2_A = N2_A + 1
    elif (De3_R_test[k] == 1):
        if (S_U_deep[k] > 0.5):
            N3_deep = N3_deep + 1
        if (S_U_L[k] > 0.5):
            N3_L = N3_L + 1
        if (S_U_A[k] > 0.5):
            N3_A = N3_A + 1


Deltas = [np.mean(De1),np.mean(De2),np.mean(De3),np.mean(De1_R_train),np.mean(De2_R_train),np.mean(De3_R_train),np.mean(De1_R_test),np.mean(De2_R_test),np.mean(De3_R_test)]
Numbers = [np.sum(De1),np.sum(De2),np.sum(De3),np.sum(De1_R_train),np.sum(De2_R_train),np.sum(De3_R_train),np.sum(De1_R_test),np.sum(De2_R_test),np.sum(De3_R_test)]

dic1 = {"tot_delta1": [Deltas[0],Numbers[0]], "tot_delta2": [Deltas[1],Numbers[1]],"tot_delta3": [Deltas[2],Numbers[2]], "train_delta1": [Deltas[3],Numbers[3]], "train_delta2": [Deltas[4],Numbers[4]],"train_delta3": [Deltas[5],Numbers[5]],"test_delta1": [Deltas[6],Numbers[6]], "test_delta2": [Deltas[7],Numbers[7]],"test_delta3": [Deltas[8],Numbers[8]]}
Result1 = pd.DataFrame(dic1,index=['rate','number'])
# ===================================================
Result1.to_csv('Result1.csv')
# ===================================================

dic2 = {"deep_delta1": N1_deep, "deep_delta2": N2_deep,"deep_delta3": N3_deep, "linear_delta1": N1_L, "linear_delta2": N2_L,"linear_delta3": N3_L, "additive_delta1": N1_A, "additive_delta2": N2_A,"additive_delta3": N3_A}
Result2 = pd.DataFrame(dic2,index=['number'])
# ===================================================
Result2.to_csv('Result2.csv')
# ===================================================

q = 1000
T_min = np.min([np.min(U_R_test),np.min(V_R_test)])
T_max = np.max([np.max(U_R_test),np.max(V_R_test)])
T_split = np.array(np.linspace(T_min, T_max, q+1), dtype="float32") + (T_max-T_min)/(2*q)
T_select = T_split[0:q]
Lamd_T_deep = I_S(m,C_D,T_select,nodevec) 
Lamd_T_L = I_S(m,C_L,T_select,nodevec) 
Lamd_T_A = I_S(m,C_A,T_select,nodevec) 

Value_IS_deep = np.zeros((len(U_R_test),q))
Value_IS_L = np.zeros((len(U_R_test),q))
Value_IS_A = np.zeros((len(U_R_test),q))

for i in range(len(U_R_test)):
    for j in range(q):
        if (De1_R_test[i] == 1):
            if (T_select[j] > U_R_test[i]):
                Value_IS_deep[i,j] = 0 - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = 0 - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = 0 - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))
            else:
                Value_IS_deep[i,j] = (np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i])) - S_U_deep[i])/(1 - S_U_deep[i]) - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = (np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i])) - S_U_L[i])/(1 - S_U_L[i]) - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = (np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i])) - S_U_A[i])/(1 - S_U_A[i]) - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))
        elif (De2_R_test[i] == 1):
            if (T_select[j] <= U_R_test[i]):
                Value_IS_deep[i,j] = 1 - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = 1 - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = 1 - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))
            elif (T_select[j] > V_R_test[i]):
                Value_IS_deep[i,j] = 0 - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = 0 - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = 0 - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))
            else:
                Value_IS_deep[i,j] = (np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i])) - S_V_deep[i])/(S_U_deep[i] - S_V_deep[i]) - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = (np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i])) - S_V_L[i])/(S_U_L[i] - S_V_L[i]) - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = (np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i])) - S_V_A[i])/(S_U_A[i] - S_V_A[i]) - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))
        elif (De3_R_test[i] == 1):
            if (T_select[j] <= V_R_test[i]):
                Value_IS_deep[i,j] = 1 - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = 1 - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = 1 - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))
            else:
                Value_IS_deep[i,j] = np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i])) / S_V_deep[i] - np.exp(- Lamd_T_deep[j] * np.exp(B_g_deep[i]))
                Value_IS_L[i,j] = np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i])) / S_V_L[i] - np.exp(- Lamd_T_L[j] * np.exp(B_g_L[i]))
                Value_IS_A[i,j] = np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i])) / S_V_A[i] - np.exp(- Lamd_T_A[j] * np.exp(B_g_A[i]))

IBS_deep = np.mean(Value_IS_deep**2)*(T_max-T_min)/T_max
IBS_L = np.mean(Value_IS_L**2)*(T_max-T_min)/T_max
IBS_A = np.mean(Value_IS_A**2)*(T_max-T_min)/T_max

dic3 = {"deep_IBS": IBS_deep, "Linear_IBS": IBS_L, "Additive_IBS": IBS_A}
Result3 = pd.DataFrame(dic3,index=['IBS'])
# ===================================================
Result3.to_csv('Result3.csv')
# ===================================================


#----------Deep-----------
B_g_D1 = np.mean(Beta_g_D1,axis=0)
B_g_D0 = np.mean(Beta_g_D0,axis=0) 
print('B_g_D1=', B_g_D1)
print('B_g_D0=', B_g_D0)
#----------Linear-----------
B_g_L1 = np.mean(Beta_g_L1,axis=0) 
B_g_L0 = np.mean(Beta_g_L0,axis=0) 
print('B_g_L1=', B_g_L1)
print('B_g_L0=', B_g_L0)
#----------Additive-----------
B_g_A1 = np.mean(Beta_g_A1,axis=0) 
B_g_A0 = np.mean(Beta_g_A0,axis=0) 
print('B_g_A1=', B_g_A1)
print('B_g_A0=', B_g_A0)

#%% Prediction of survival function for subjects
# Calculate and draw three graphs with delta3 = 1
for k in range(len(V1)):
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1, 1, 1)
    ax6.set_xlabel("t",fontsize=8)     
    ax6.set_ylabel(r'$\hat{S}(t)$',fontsize=8) 
    ax6.tick_params(axis='both',labelsize=6) 
    ax6.xaxis.set_ticks_position('bottom')
    ax6.spines['bottom'].set_position(('data',0))
    ax6.yaxis.set_ticks_position('left')
    ax6.spines['left'].set_position(('data',0))
    ax6.grid(True)
    # Calculate S(t)
    St_D1 = np.exp(-I_S(m,C_D,V1_value,nodevec) * np.exp(B_g_D1[k]))
    
    St_L1 = np.exp(-I_S(m,C_L,V1_value,nodevec) * np.exp(B_g_L1[k]))
    
    St_A1 = np.exp(-I_S(m,C_A,V1_value,nodevec) * np.exp(B_g_L1[k]))
    
    ax6.plot(V1_value, St_D1, color='blue', linestyle='--')
    ax6.plot(V1_value, St_L1, color='orange', linestyle=':')
    ax6.plot(V1_value, St_A1, color='green', linestyle='-.')
    ax6.plot(V1_value, 0.5*np.ones(len(V1_value)), color='red', linestyle='-')
    
    if (k==0):
        ax6.plot(V1_015, np.exp(-I_S(m,C_D,np.array([V1_015]),nodevec) * np.exp(B_g_D1[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax6.plot(V1_015, np.exp(-I_S(m,C_L,np.array([V1_015]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax6.plot(V1_015, np.exp(-I_S(m,C_A,np.array([V1_015]),nodevec) * np.exp(B_g_L1[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax6.plot(np.array([V1_015,V1_015]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_015]),nodevec) * np.exp(B_g_D1[k])), np.exp(-I_S(m,C_L,np.array([V1_015]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([V1_015]),nodevec) * np.exp(B_g_L1[k]))])], dtype='float32'), color='k', linestyle=':')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$(b)\quad\Delta_3=1, 15^{\rm{th}}$', fontsize=10) # Set title and size
        fig6.savefig('fig1_15.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax6.plot(V1_030, np.exp(-I_S(m,C_D,np.array([V1_030]),nodevec) * np.exp(B_g_D1[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax6.plot(V1_030, np.exp(-I_S(m,C_L,np.array([V1_030]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax6.plot(V1_030, np.exp(-I_S(m,C_A,np.array([V1_030]),nodevec) * np.exp(B_g_L1[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax6.plot(np.array([V1_030,V1_030]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_030]),nodevec) * np.exp(B_g_D1[k])), np.exp(-I_S(m,C_L,np.array([V1_030]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([V1_030]),nodevec) * np.exp(B_g_L1[k]))])], dtype='float32'), color='k', linestyle=':')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$(d)\quad\Delta_3=1, 30^{\rm{th}}$', fontsize=10) # Set title and size
        fig6.savefig('fig1_30.jpeg', dpi=400, bbox_inches='tight')
    elif (k==2):
        ax6.plot(V1_045, np.exp(-I_S(m,C_D,np.array([V1_045]),nodevec) * np.exp(B_g_D1[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax6.plot(V1_045, np.exp(-I_S(m,C_L,np.array([V1_045]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax6.plot(V1_045, np.exp(-I_S(m,C_A,np.array([V1_045]),nodevec) * np.exp(B_g_L1[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax6.plot(np.array([V1_045,V1_045]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_045]),nodevec) * np.exp(B_g_D1[k])), np.exp(-I_S(m,C_L,np.array([V1_045]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([V1_045]),nodevec) * np.exp(B_g_L1[k]))])], dtype='float32'), color='k', linestyle=':')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$(f)\quad\Delta_3=1, 45^{\rm{th}}$', fontsize=10) # Set title and size
        fig6.savefig('fig1_45.jpeg', dpi=400, bbox_inches='tight')
    elif (k==3):
        ax6.plot(V1_060, np.exp(-I_S(m,C_D,np.array([V1_060]),nodevec) * np.exp(B_g_D1[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax6.plot(V1_060, np.exp(-I_S(m,C_L,np.array([V1_060]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax6.plot(V1_060, np.exp(-I_S(m,C_A,np.array([V1_060]),nodevec) * np.exp(B_g_L1[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax6.plot(np.array([V1_060,V1_060]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_060]),nodevec) * np.exp(B_g_D1[k])), np.exp(-I_S(m,C_L,np.array([V1_060]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([V1_060]),nodevec) * np.exp(B_g_L1[k]))])], dtype='float32'), color='k', linestyle=':')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$(h)\quad\Delta_3=1, 60^{\rm{th}}$', fontsize=10) # Set title and size
        fig6.savefig('fig1_60.jpeg', dpi=400, bbox_inches='tight')
    elif (k==4):
        ax6.plot(V1_075, np.exp(-I_S(m,C_D,np.array([V1_075]),nodevec) * np.exp(B_g_D1[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax6.plot(V1_075, np.exp(-I_S(m,C_L,np.array([V1_075]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax6.plot(V1_075, np.exp(-I_S(m,C_A,np.array([V1_075]),nodevec) * np.exp(B_g_L1[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax6.plot(np.array([V1_075,V1_075]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_075]),nodevec) * np.exp(B_g_D1[k])), np.exp(-I_S(m,C_L,np.array([V1_075]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([V1_075]),nodevec) * np.exp(B_g_L1[k]))])], dtype='float32'), color='k', linestyle=':')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$(j)\quad\Delta_3=1, 75^{\rm{th}}$', fontsize=10) # Set title and size
        fig6.savefig('fig1_75.jpeg', dpi=400, bbox_inches='tight')
        
# Calculate and draw three figures with delta3=0
for k in range(len(V1)):
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(1, 1, 1)
    ax7.set_xlabel("t",fontsize=8)    
    ax7.set_ylabel(r'$\hat{S}(t)$',fontsize=8) 
    ax7.tick_params(axis='both',labelsize=6) 
    ax7.xaxis.set_ticks_position('bottom')
    ax7.spines['bottom'].set_position(('data',0))
    ax7.yaxis.set_ticks_position('left')
    ax7.spines['left'].set_position(('data',0))
    ax7.grid(True)
    
    St_D0 = np.exp(-I_S(m,C_D,V0_value,nodevec) * np.exp(B_g_D0[k]))
    
    St_L0 = np.exp(-I_S(m,C_L,V0_value,nodevec) * np.exp(B_g_L0[k]))
    
    St_A0 = np.exp(-I_S(m,C_A,V0_value,nodevec) * np.exp(B_g_A0[k]))
    
    ax7.plot(V0_value, St_D0, color='blue', linestyle='--')
    ax7.plot(V0_value, St_L0, color='orange', linestyle=':')
    ax7.plot(V0_value, St_A0, color='green', linestyle='-.')
    ax7.plot(V0_value, 0.5*np.ones(len(V0_value)), color='red', linestyle='-')
    
    if (k==0):
        ax7.plot(V0_015, np.exp(-I_S(m,C_D,np.array([V0_015]),nodevec) * np.exp(B_g_D0[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax7.plot(V0_015, np.exp(-I_S(m,C_L,np.array([V0_015]),nodevec) * np.exp(B_g_L0[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax7.plot(V0_015, np.exp(-I_S(m,C_A,np.array([V0_015]),nodevec) * np.exp(B_g_A0[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax7.plot(np.array([V0_015,V0_015]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_015]),nodevec) * np.exp(B_g_D0[k])), np.exp(-I_S(m,C_L,np.array([V0_015]),nodevec) * np.exp(B_g_L0[k])), np.exp(-I_S(m,C_A,np.array([V0_015]),nodevec) * np.exp(B_g_A0[k]))])], dtype='float32'), color='k', linestyle=':')
        ax7.legend(loc='best', fontsize=6)
        ax7.set_title(r'$(a)\quad\Delta_3=0, 15^{\rm{th}}$', fontsize=10) # Set title and size
        fig7.savefig('fig0_15.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax7.plot(V0_030, np.exp(-I_S(m,C_D,np.array([V0_030]),nodevec) * np.exp(B_g_D0[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax7.plot(V0_030, np.exp(-I_S(m,C_L,np.array([V0_030]),nodevec) * np.exp(B_g_L0[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax7.plot(V0_030, np.exp(-I_S(m,C_A,np.array([V0_030]),nodevec) * np.exp(B_g_A0[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax7.plot(np.array([V0_030,V0_030]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_030]),nodevec) * np.exp(B_g_D0[k])), np.exp(-I_S(m,C_L,np.array([V0_030]),nodevec) * np.exp(B_g_L0[k])), np.exp(-I_S(m,C_A,np.array([V0_030]),nodevec) * np.exp(B_g_A0[k]))])], dtype='float32'), color='k', linestyle=':')
        ax7.legend(loc='best', fontsize=6)
        ax7.set_title(r'$(c)\quad\Delta_3=0, 30^{\rm{th}}$', fontsize=10) # Set title and size
        fig7.savefig('fig0_30.jpeg', dpi=400, bbox_inches='tight')
    elif (k==2):
        ax7.plot(V0_045, np.exp(-I_S(m,C_D,np.array([V0_045]),nodevec) * np.exp(B_g_D0[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax7.plot(V0_045, np.exp(-I_S(m,C_L,np.array([V0_045]),nodevec) * np.exp(B_g_L0[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax7.plot(V0_045, np.exp(-I_S(m,C_A,np.array([V0_045]),nodevec) * np.exp(B_g_A0[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax7.plot(np.array([V0_045,V0_045]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_045]),nodevec) * np.exp(B_g_D0[k])), np.exp(-I_S(m,C_L,np.array([V0_045]),nodevec) * np.exp(B_g_L0[k])), np.exp(-I_S(m,C_A,np.array([V0_045]),nodevec) * np.exp(B_g_A0[k]))])], dtype='float32'), color='k', linestyle=':')
        ax7.legend(loc='best', fontsize=6)
        ax7.set_title(r'$(e)\quad\Delta_3=0, 45^{\rm{th}}$', fontsize=10) # Set title and size
        fig7.savefig('fig0_45.jpeg', dpi=400, bbox_inches='tight')
    elif (k==3):
        ax7.plot(V0_060, np.exp(-I_S(m,C_D,np.array([V0_060]),nodevec) * np.exp(B_g_D0[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax7.plot(V0_060, np.exp(-I_S(m,C_L,np.array([V0_060]),nodevec) * np.exp(B_g_L0[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax7.plot(V0_060, np.exp(-I_S(m,C_A,np.array([V0_060]),nodevec) * np.exp(B_g_A0[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax7.plot(np.array([V0_060,V0_060]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_060]),nodevec) * np.exp(B_g_D0[k])), np.exp(-I_S(m,C_L,np.array([V0_060]),nodevec) * np.exp(B_g_L0[k])), np.exp(-I_S(m,C_A,np.array([V0_060]),nodevec) * np.exp(B_g_A0[k]))])], dtype='float32'), color='k', linestyle=':')
        ax7.legend(loc='best', fontsize=6)
        ax7.set_title(r'$(g)\quad\Delta_3=0, 60^{\rm{th}}$', fontsize=10) # Set title and size
        fig7.savefig('fig0_60.jpeg', dpi=400, bbox_inches='tight')
    elif (k==4):
        ax7.plot(V0_075, np.exp(-I_S(m,C_D,np.array([V0_075]),nodevec) * np.exp(B_g_D0[k])), label='DNN-based', marker='o', markersize=4, ls='--', color='blue')
        ax7.plot(V0_075, np.exp(-I_S(m,C_L,np.array([V0_075]),nodevec) * np.exp(B_g_L0[k])), label='CPH', marker='s', markersize=4, ls=':', color='orange')
        ax7.plot(V0_075, np.exp(-I_S(m,C_A,np.array([V0_075]),nodevec) * np.exp(B_g_A0[k])), label='PLACM', marker='^', markersize=4, ls='-.', color='green')
        ax7.plot(np.array([V0_075,V0_075]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_075]),nodevec) * np.exp(B_g_D0[k])), np.exp(-I_S(m,C_L,np.array([V0_075]),nodevec) * np.exp(B_g_L0[k])), np.exp(-I_S(m,C_A,np.array([V0_075]),nodevec) * np.exp(B_g_A0[k]))])], dtype='float32'), color='k', linestyle=':')
        ax7.legend(loc='best', fontsize=6)
        ax7.set_title(r'$(i)\quad\Delta_3=0, 75^{\rm{th}}$', fontsize=10) # Set title and size
        fig7.savefig('fig0_75.jpeg', dpi=400, bbox_inches='tight')