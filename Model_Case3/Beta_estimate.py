
import numpy as np
import scipy.optimize as spo

def Beta_est(De1, De2, De3, Z, Lambda_U, Lambda_V, g_X):
    def BF(*args):
        Ezg = np.exp(Z * args[0] + g_X)
        Loss_F = - np.mean(De1 * np.log(1 - np.exp(- Lambda_U * Ezg) + 1e-5) + De2 * np.log(np.exp(- Lambda_U * Ezg) - np.exp(- Lambda_V * Ezg) + 1e-5) - De3 * Lambda_V * Ezg)
        return Loss_F
    result = spo.minimize(BF,np.zeros(1),method='SLSQP')
    return result['x']
