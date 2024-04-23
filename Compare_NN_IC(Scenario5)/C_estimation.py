import numpy as np
import scipy.optimize as spo
from I_spline import I_U
def C_est(m, U, V, De1, De2, De3, g_X, nodevec):
    Iu = I_U(m, U, nodevec)
    Iv = I_U(m, V, nodevec)
    def LF(*args):
        a = args[0]
        Ezg = np.exp(g_X)
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu,a) * Ezg) + 1e-5) + De2 * np.log(np.exp(- np.dot(Iu,a) * Ezg) - np.exp(- np.dot(Iv,a) * Ezg) + 1e-5) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = []
    for i in range(m+3):
        bnds.append((0,100))
    result = spo.minimize(LF,np.ones(m+3),method='SLSQP',bounds=bnds)
    return result['x']


