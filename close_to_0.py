#Import the file that calculates the matrix T_tilde
import main

#Import the needed librairies
import numpy as np
import math
import matplotlib.pyplot as plt
from random import seed, randint
import time
import matplotlib.animation as animation
import os

eps = 0.03

def smallest_eigenvalues(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvectors_T = eigenvectors.T
    norms = [abs(z) for z in eigenvalues]
    smallest_eigenvalue = min(norms)
    indices = [i for i, num in enumerate(norms) if num == smallest_eigenvalue]
    small_values = []
    small_vectors = []
    for i in indices:
        small_values.append(eigenvalues[i])
        phase = np.angle(eigenvectors_T[i]/np.linalg.norm(eigenvectors_T[i][0]))  # Get angle (theta) of first complex entry
        small_vectors.append(np.exp(-1j*phase)*eigenvectors_T[i]/np.linalg.norm(eigenvectors_T[i]))
    return small_values,small_vectors

def print_smallest_eigenvalue(A):
    small_values,  small_vectors = smallest_eigenvalues(A)
    real_parts = np.real(small_values)
    imag_parts = np.imag(small_values)
    print('Number of smallest eigenvalues = '+str(len(small_values)))
    print('Eigenvectors associated to each eigenvalues :')
    for i in range(len(small_vectors)):
        print('Eigenvalue '+str(small_values[i])+' with '+str(small_vectors[i])+' Eigenvector')
    plt.figure(figsize=(6, 6))
    plt.scatter(real_parts, imag_parts, color='blue', marker='o', label='Smallest eigenvalues',s=50,alpha=0.2)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Smallest eigenvalue(s) ('+str(len(small_values))+' value(s)) in the Complex Plane for '+r'$\tilde{T}$'+f' as a size '+str(np.shape(A)[0])+' matrix.')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.show()

def eigenvector_deformation(eigenvector,theta,shape):
    res = 0
    for i in range(len(eigenvector)):
        res += eigenvector[i]*math.cos(2*math.pi*(i+1)*main.x_lazutkin(theta,shape))
    return res

def gamma_deformation(theta,shape,vector,delta):
    add1 = (eigenvector_deformation(vector,theta,shape)*-math.sin(theta),eigenvector_deformation(vector,theta,shape)*math.cos(theta))
    add = (delta*add1[0],delta*add1[1])
    res = (main.gamma(theta,shape)[0] + add[0],main.gamma(theta,shape)[1] + add[1])
    return res
    

if __name__ == '__main__':
    tronc = 20
    #omega = np.array([1])
    omega = np.array([1 ,0, 0.5, -0.45, 0.3])
    #omega = np.array([1 ,0, 0, 0.050])
    theta = np.linspace(0, 2 * np.pi, 5000)
    T = main.T_matrix(omega,tronc)
    T_tilde = main.tilde(T)
    abs_deformation_values = [abs(eigenvector_deformation(smallest_eigenvalues(T_tilde)[1][0],t,omega)) for t in theta]
    max_abs_deformation_values = max(abs_deformation_values)
    if max_abs_deformation_values != 0:
        delta = eps/max_abs_deformation_values
    else:
        print("Deformation is 0 !!!")
        delta = eps
    #smallest_eigenvalues(T_tilde)
    x_vals, y_vals = np.array([main.gamma(t,omega) for t in theta]).T
    x_vals1, y_vals1 = np.array([gamma_deformation(t,omega,smallest_eigenvalues(T_tilde)[1][0],delta) for t in theta]).T
    print(np.size(np.array([gamma_deformation(t,omega,smallest_eigenvalues(T_tilde)[1][0],delta) for t in theta]).T))
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(x_vals, y_vals, label="Shape", color='blue')
    plt.plot(x_vals1, y_vals1, label="Deformed", color='red',linestyle='dashed')
    ## Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Deformed shape")
    # Add grid and legend
    plt.legend()
    plt.axis("equal")  # Ensure correct aspect ratio
    plt.grid(True)
    plt.show()
    end = time.time()
    print(f'Total time with T of size {tronc} is {end - main.start:.4f} seconds')
    #print_smallest_eigenvalue(T_tilde)