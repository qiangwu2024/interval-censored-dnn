import numpy as np
import scipy.optimize as spo
def g_L(train_data,X_test,Lambda_U,Lambda_V,Beta0):
    Z_train = train_data['Z']
    X_train = train_data['X']
    De1 = train_data['De1']
    De2 = train_data['De2']
    De3 = train_data['De3']
    def GF(*args):
        b = args[0]
        Ezg = np.exp(Z_train * Beta0 + np.dot(X_train,b[0:5]) + b[5]*np.ones(X_train.shape[0]))
        loss_fun = - np.mean(De1 * np.log(1 - np.exp(- Lambda_U * Ezg) + 1e-5) + De2 * np.log(np.exp(- Lambda_U * Ezg) - np.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return loss_fun
    linear_para = spo.minimize(GF,np.zeros(6),method='SLSQP')['x']
    g_train = np.dot(X_train,linear_para[0:5]) + linear_para[5]*np.ones(X_train.shape[0])
    g_test = np.dot(X_test,linear_para[0:5]) + linear_para[5]*np.ones(X_test.shape[0])
    return {'linear_para': linear_para,
        'g_train': g_train,
        'g_test': g_test
    }
