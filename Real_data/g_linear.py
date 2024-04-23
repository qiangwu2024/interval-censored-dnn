import numpy as np
import scipy.optimize as spo
def g_L(Z_train,X_train,De1,De2,De3,X_test,X_sort1,X_sort0,Lambda_U,Lambda_V):
    c = Z_train.shape[1]
    d = X_train.shape[1]
    def GF(*args):
        b = args[0]
        Ezg = np.exp(np.dot(Z_train, b[(d+1):(d+c+1)]) + np.dot(X_train,b[0:d]) + b[d]*np.ones(X_train.shape[0]))
        loss_fun = - np.mean(De1 * np.log(1 - np.exp(- Lambda_U * Ezg) + 1e-5) + De2 * np.log(np.exp(- Lambda_U * Ezg) - np.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return loss_fun
    linear_para = spo.minimize(GF,np.zeros(d+c+1),method='SLSQP')['x']
    print('linear_parameter=', linear_para)
    g_train = np.dot(X_train,linear_para[0:d]) + linear_para[d]*np.ones(X_train.shape[0])
    g_test = np.dot(X_test,linear_para[0:d]) + linear_para[d]*np.ones(X_test.shape[0])
    g_test1 = np.dot(X_sort1,linear_para[0:d]) + linear_para[d]*np.ones(X_sort1.shape[0])
    g_test0 = np.dot(X_sort0,linear_para[0:d]) + linear_para[d]*np.ones(X_sort0.shape[0])
    
    return {'linear_para': linear_para,
        'g_train': g_train,
        'g_test': g_test,
        'g_test1': g_test1,
        'g_test0': g_test0,
        'Beta': linear_para[(d+1):(d+c+1)] 
    }