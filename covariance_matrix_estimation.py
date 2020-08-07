import numpy as np
import matplotlib.pyplot as plt 

#set the range and values of z
def z_sample(z_re_min, z_re_max, z_re_num, z_im_min, z_im_max, z_im_num):
    z_re = np.linspace(z_re_min, z_re_max, z_re_num, True, False, dtype=float)
    z_im = np.linspace(z_im_min, z_im_max, z_im_num,True, False, dtype=float)
    z = []
    z_num = z_re_num * z_im_num

    for i in range(z_re_num):
        for j in range(z_im_num):
            z_ij = complex(z_re[i], z_im[j])
            z.append(z_ij)
    return z, z_num

#Stieltjes transform
def ST(z, z_num, X):
    m_F_p = np.empty(z_num, dtype='complex')
    v_F_p = np.empty(z_num, dtype='complex')

    S_p = (1 / n) * np.dot(X.T, X) #compute the covariance matrix

    for i in range(z_num):
        m_F_p[i] = (1/ p) * np.trace(S_p - z[i] * np.eye(p, dtype=float))   #compute the stieltjes transform m_F_p
        v_F_p[i] = (gamma - 1) * (1 / z[i]) + gamma * m_F_p[i] #the stieltjes transform v_F_p

    return v_F_p

#error function
def error(z, z_num, X, w):
    
    e = np.empty(z_num, dtype='complex')
    Re_e = np.empty(z_num)
    Im_e = np.empty(z_num)

    v_F_p = ST(z, z_num, X)

    for i in range(z_num):
        e[i] = 1 / v_F_p[i] + z[i] - gamma * np.sum((w * t) / (1 + v_F_p[i] * t))
        Re_e[i] = e[i].real
        Im_e[i] = e[i].imag

    Re_e_max = np.max(np.abs(Re_e))
    Im_e_max = np.max(np.abs(Im_e))
    err = np.max([Re_e_max, Im_e_max])

    return err

#compute gradient
def numerical_gradient(z, z_num, X, w):
    eps = 1e-4
    grad = np.zeros_like(w)

    for i in range(w.size):
        tmp_val = w[i]
        w[i] = tmp_val + eps 
        fxh1 = error(z, z_num, X, w)

        w[i] = tmp_val - eps
        fxh2 = error(z, z_num, X, w)

        grad[i] = (fxh1 - fxh2) / (2*eps)
        w[i] = tmp_val
    return grad

#optimization
def opt(z, z_num, X, w):
    for i in range(step_num):
        grad = numerical_gradient(z, z_num, X, w)
        for index in range(len(w)):
            w[index] -= lr * grad[index]

        non_negative_w = np.abs(w)
        sum_w = np.sum(non_negative_w)
        w = non_negative_w / sum_w
        err = error(z, z_num, X, w)
        print(err)
    
    CDF = w_acc = [w[0]]
    
    for i in range(1, k):
        w_acc = w_acc + w[i]
        CDF.append(w_acc)

    return w, CDF


if __name__ == "__main__":
    X = np.eye(500)       # X is a n * p matrix
    p = 500    #the number of variables
    n = 500      #the sample size  
    gamma = p / n 
    step_num = 100
    lr = 0.1    #learning rate
    k = 100     #number of t
    w_init = np.random.uniform(low=0, high=1, size=k)   #initial w

    t_k_min = (1 - np.sqrt(gamma)) ** 2     #minimum of t
    t_k_max =  (1 + np.sqrt(gamma)) ** 2    #maximum of t
    t = np.linspace(t_k_min, t_k_max, k, True, False, dtype=float)

    z, z_num = z_sample(0, 1, 50, -0.001, -0.01, 50)     
    w, CDF = opt(z, z_num, X, w_init)
 
    plt.plot(t, CDF)
    plt.xlabel('t')
    plt.ylabel('CDF')
    plt.show()



















