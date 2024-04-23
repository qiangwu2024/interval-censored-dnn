
import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_I(n, corr, Beta, tau):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(5)
    cov = np.identity(5) * (1-corr) + np.ones((5, 5)) * corr
    X = ndm.multivariate_normal(mean, cov, n)
    X = np.clip(X, 0, 2)
    g_X =  X[:,0]/2 + X[:,1] / 3 + X[:,2] / 3 + X[:,3] / 4 + X[:,4] / 5 - 0.63
    Y = ndm.rand(n)
    T = (- 5 * np.log(Y) * np.exp(- Z * Beta - g_X)) ** 2
    U = uniform_data(n, 0, tau / 10)
    V_0 = tau / 5 + U + ndm.exponential(1, n) * tau / 2
    V = np.clip(V_0, 0, tau)
    De1 = (T <= U)
    De2 = (U < T) * (T <= V)
    De3 = (T > V)
    return {
        'Z': np.array(Z, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'V': np.array(V, dtype='float32'),
        'De1': np.array(De1, dtype='float32'),
        'De2': np.array(De2, dtype='float32'),
        'De3': np.array(De3, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32')
    }
