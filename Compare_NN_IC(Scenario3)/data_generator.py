import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_scenario_3(n, p_dim, lambda_param, k_param, tau):
    mean = np.zeros(p_dim) 
    rho = np.exp(1) 
    def covariance(p):
        cov = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                cov[i, j] = rho**(-abs(i - j))
        return cov
    Z = ndm.multivariate_normal(mean, covariance(p_dim), n)
    Z[:,np.arange(int(0.2*p_dim),int(0.4*p_dim))] = (Z[:,np.arange(int(0.2*p_dim),int(0.4*p_dim))]>0).astype(int) 
    Z[:,np.arange(int(0.4*p_dim), p_dim)] = (Z[:,np.arange(int(0.4*p_dim), p_dim)]>-0.5).astype(int) + (Z[:,np.arange(int(0.4*p_dim), p_dim)]>0.5).astype(int)

    beta = 0.2*np.ones(p_dim)
    beta[np.arange(int(0.4*p_dim), p_dim)] = ndm.multivariate_normal(0.2*np.ones(int(0.6*p_dim)), 0.01*covariance(int(0.6*p_dim)))
    g_Z = np.dot(Z, beta) + Z[:,2] * Z[:,3]
    Y = ndm.rand(n)
    T = (-np.log(Y)* np.exp(- g_Z))**(1/k_param) / lambda_param
    U_0 = ndm.exponential(1, n) * tau / 6
    U = np.clip(U_0, 0, 5*tau/6)
    V_0 = tau / 3 + U + ndm.exponential(1, n) * tau / 4
    V = np.clip(V_0, 0, tau)
    De1 = (T <= U)
    De2 = (U < T) * (T <= V)
    De3 = (T > V)
    return {
        'Z': np.array(Z, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'V': np.array(V, dtype='float32'),
        'De1': np.array(De1, dtype='float32'),
        'De2': np.array(De2, dtype='float32'),
        'De3': np.array(De3, dtype='float32'),
        'g_Z': np.array(g_Z, dtype='float32')
    }
