#Import the file that calculates the matrix T_tilde
import main
import close_to_0

#Import the needed librairies
import numpy as np
import math
import matplotlib.pyplot as plt
from random import seed, randint
import time
import matplotlib.animation as animation
import os

close_to_0.eps = 0.005

def l_q_deformation(q,T):
    size = np.shape(T)[0]
    index = q - 1
    elementary_vector = np.eye(1,size,index).T
    deformation_vector = np.dot(np.linalg.inv(T),elementary_vector)
    #print(np.dot(T,deformation_vector))
    res = deformation_vector.flatten().tolist()
    return res

if __name__ == '__main__':
    tronc = 20
    q = 4
    #omega = np.array([1])
    omega = np.array([1 ,0, 0.5, -0.45, 0.3])
    #omega = np.array([1 ,0, 0, 0.99])
    theta = np.linspace(0, 2 * np.pi, 500)
    T = main.T_matrix(omega,tronc)
    #print(T)
    T_tilde = main.tilde(T)

    abs_deformation_values = [abs(close_to_0.eigenvector_deformation(close_to_0.smallest_eigenvalues(T_tilde)[1][0],t,omega)) for t in theta]
    max_abs_deformation_values = max(abs_deformation_values)
    if max_abs_deformation_values != 0:
        delta = close_to_0.eps/max_abs_deformation_values
    else:
        print("Deformation is 0 !!!")
        delta = close_to_0.eps

    theta0 = main.find_theta0_root(q, omega) # Find theta0 using Root's method
    #theta0 = main.find_theta0_newton(q,omega) # Find theta0 using Newton's method
    #theta0 = [(2*math.pi)*(i/q) for i in range(q)] #Uniform theta0
    theta_impacts = main.L_maximizer(theta0, omega)
    ksi = np.array([main.x_lazutkin(theta, omega) for theta in theta_impacts])  # Ensure ksi is a numpy array
    impacts = main.find_impacts(theta_impacts, omega)

    impacts.append(impacts[0])
    x_vals2, y_vals2 = np.array(impacts).T
    plt.figure(figsize=(7.5, 3.5))
    x_vals, y_vals = np.array([main.gamma(t,omega) for t in theta]).T
    x_vals1, y_vals1 = np.array([close_to_0.gamma_deformation(t,omega,l_q_deformation(q,T),delta) for t in theta]).T
    plt.plot(x_vals, y_vals, label="Shape", color='blue')
    plt.plot(x_vals1, y_vals1, label="Deformed shape", color='red',linestyle='dashed')
    plt.plot(x_vals2, y_vals2, label="1/q Trajectory", color='red',linestyle='-', alpha = 0.6)
    plt.scatter(x_vals2, y_vals2, label="1/q Impacts", color='red', marker='o', s=50, alpha = 0.6)
    del impacts[-1]

    for j in range(max(2,q-2),q+2):
        if j == q:
            pass
        else:
            theta0_2 = main.find_theta0_root(j, omega) # Find theta0 using Root's method
            #theta0_2 = main.find_theta0_newton(j,omega) # Find theta0 using Newton's method
            #theta0_2 = [(2*math.pi)*(i/q) for i in range(j)] #Uniform theta0
            theta_impacts_2 = main.L_maximizer(theta0_2, omega)
            ksi_2 = np.array([main.x_lazutkin(theta, omega) for theta in theta_impacts_2])  # Ensure ksi is a numpy array
            impacts_2 = main.find_impacts(theta_impacts_2, omega)
            impacts_2.append(impacts_2[0])
            x_valsj, y_valsj = np.array(impacts_2).T
            plt.plot(x_valsj, y_valsj, color='green',linestyle='-',alpha = 0.2)
            plt.scatter(x_valsj, y_valsj, color='green', marker='o', s=50,alpha=0.2)
            del impacts_2[-1]
    
    ## Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    ell = '\u2113'  # ℓ with subscript 1
    end = time.time()
    plt.title("Tronc ="+str(tronc)+r',  $\epsilon$ ='+str(close_to_0.eps)+", "+r'$\ell_{q}$'+" deformation for q="+str(q)+'. Running time'f' = {(end - main.start)/60:.4f} min')
    # Add grid and legend
    plt.legend()
    plt.axis("equal")  # Ensure correct aspect ratio
    plt.grid(True)
    plt.savefig(str(omega)+'_l_q_changes_q='+str(q)+'_tronc='+str(tronc)+'.png',dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Total time with T of size {tronc} is {end - main.start:.4f} seconds')