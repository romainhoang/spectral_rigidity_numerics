import main 
import close_to_0

import math
import numpy as np
import matplotlib.pyplot as plt
from random import*
import random
from scipy.optimize import minimize, root_scalar, newton
import scipy.integrate as integrate
import time

# Start the timer
start = time.time()

tronc = 10

def grad_omega(omega):
    h = 0.001
    res = []
    for i in range(2,len(omega)):
        T = abs(close_to_0.smallest_eigenvalues(main.tilde(main.T_matrix(omega,tronc)))[0][0])
        omega[i] += h
        T_h = abs(close_to_0.smallest_eigenvalues(main.tilde(main.T_matrix(omega,tronc)))[0][0])
        omega[i] += -h
        res.append(float(((T_h - T)/h)))
    return res

eps = 0.001
def schema(omega,iter):
    val = []
    x_iter = [i for i in range(iter)]
    theta = np.linspace(0, 2 * np.pi, 5000)
    x_vals, y_vals = np.array([main.gamma(t,omega) for t in theta]).T
    Y = [main.rho_function(x,omega)  for x in theta]
    for i in range(iter):
        print('i = '+str(i))
        d_omega = grad_omega(omega)
        omega[2:] = omega[2:] - eps*np.array(d_omega)
        val.append(abs(close_to_0.smallest_eigenvalues(main.tilde(main.T_matrix(omega,tronc)))[0][0]))
    x_vals_1, y_vals_1 = np.array([main.gamma(t,omega) for t in theta]).T
    Z = [main.rho_function(x,omega)  for x in theta]
    v = abs(close_to_0.smallest_eigenvalues(main.tilde(main.T_matrix(omega,tronc)))[0][0])
    print('absolute value of the smallest eigenvalue : '+str(v))
    plt.figure(figsize=(7.5, 3.5))
    plt.scatter(x_iter,val, label="Smallest eigenvalue")
    plt.title("Absolute value of the smallest eigenvalue at each step of the scheme")
    plt.xlabel('i')
    plt.ylabel(r"$|\lambda_{\min}|$")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(x_vals, y_vals, label="Initial shape", color='blue')
    plt.plot(x_vals_1, y_vals_1, label=f"New shape after {iter}-iterations of the schema ", color='red',linestyle='dashed')
    plt.title('Shape, absolute value of the smallest eigenvalue (for new shape) = '+str(v))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    #plt.axis('equal')
    plt.legend()
    plt.show()
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(theta, Y, label=r"Initial $\rho(\theta)$ ", color='blue')
    plt.plot(theta, Z, label=r'New $\rho(\theta)$ '+f" after {iter}-iterations of the schema ", color='red',linestyle='dashed')
    plt.title(r'radius of curvature $\rho(\theta)$, absolute value of the smallest eigenvalue (for new shape) = '+str(v))
    plt.ylabel(r'$\rho(\theta)$')
    plt.xlabel('x')
    plt.grid(True)
    #plt.axis('equal')
    plt.legend()
    plt.show()
    return omega

def plot_shape(shape):
    theta = np.linspace(0, 2 * np.pi, 5000)
    x_vals, y_vals = np.array([main.gamma(t,shape) for t in theta]).T
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(x_vals, y_vals, label="Shape", color='blue')
    plt.title('Shape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


if __name__=='__main__':
    omega = np.array([1 ,0, 0.5, -0.45, 0.3])
    #omega = [1, 0, 0.1, -0.2, 0.3, -0.1, 0.1, 0.1, 0.3, 0.1]
    #omega = [1,0,0.2]
    #omega = [1, 0, 0.2, -0.75]
    #omega = [1]
    schema(omega,150)
    end = time.time()
    print('Total running time of the code :'+f'{end - start:.4f} seconds = {(end - start)/60:.4f} minutes')

        