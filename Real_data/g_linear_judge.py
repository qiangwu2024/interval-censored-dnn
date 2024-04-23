import numpy as np
import scipy.optimize as spo
def g_L_judge(Z_train,X_train,De1,De2,De3,Lambda_U,Lambda_V):
    c = Z_train.shape[1]
    d = X_train.shape[1]
    def GF(*args):
        b = args[0]
        Ezg = np.exp(np.dot(Z_train, b[(d+1):(d+c+1)]) + np.dot(X_train,b[0:d]) + b[d]*np.ones(X_train.shape[0]))
        loss_fun = - np.mean(De1 * np.log(1 - np.exp(- Lambda_U * Ezg) + 1e-5) + De2 * np.log(np.exp(- Lambda_U * Ezg) - np.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return loss_fun
    linear_para = spo.minimize(GF,np.zeros(d+c+1),method='SLSQP')['x']
    print('linear_parameter=', linear_para)
    g_train = np.dot(X_train,linear_para[0:d]) + linear_para[d]*np.ones(X_train.shape[0])+np.dot(Z_train[:,4:c], linear_para[(d+c-2):(d+c+1)])
    
    return {'linear_para': linear_para[0:d],
        'g_train': g_train,
        'Beta': linear_para[(d+1):(d+c+1)] 
    }