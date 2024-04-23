import numpy as np
import scipy.optimize as spo

def Beta_est(De1, De2, De3, Z, Lambda_U, Lambda_V, g_X):
    def BF(*args):
        a = args[0]
        Ezg = np.exp(np.dot(Z, a) + g_X)
        Loss = - np.mean(De1 * np.log(1 - np.exp(- Lambda_U * Ezg) + 1e-5) + De2 * np.log(np.exp(- Lambda_U * Ezg) - np.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return Loss
    result = spo.minimize(BF,np.zeros(Z.shape[1]),method='SLSQP')
    return result['x']
