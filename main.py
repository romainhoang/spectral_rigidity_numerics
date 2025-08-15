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

#### GENERATING A DOMAIN
#### GENERATING A DOMAIN


# Set the seed for random number generation
seed_value = 3

#Generating random sequence of Fourier coefficients normalized, such as sum(a2,...)<1 and ai<1/nbr_coeff
def rho_coeff(nbr_coeff):
    seed(seed_value)
    res = [nbr_coeff,0]
    if nbr_coeff <= 2:
        return [1,0]
    else:
        for i in range(nbr_coeff-2):
            n = random.randint(-10**3,10**3)/10**3
            res.append(n)
    res = [x / nbr_coeff for x in res]
    return res

#Generating the same random sequence of Fourier coefficients non normalized
def rho_coeff_non_normalized(nbr_coeff):
    seed(seed_value)
    res = [1,0]
    if nbr_coeff <= 2:
        return [1,0]
    else:
        for i in range(nbr_coeff-2):
            n = random.randint(-10**3,10**3)/10**3
            res.append(n)
    return res

#Defining the function rho as a troncature of a Fourier Serie
def rho_function(theta,shape):
    y = 0
    for j in range(len(shape)):
        y += shape[j]*math.cos(j*theta)
    return y

#To plot rho over [0,2*pi]
def plot_rho(shape):
    X = np.linspace(0,2*math.pi,1000)
    Y = [rho_function(x,shape)  for x in X]
    Z = np.zeros(len(X))
    plt.figure()
    plt.title(r'radius of curvature $\rho(\theta)$')
    plt.ylabel(r'$\rho(\theta)$')
    plt.xlabel('x')
    plt.plot(X,Z,)
    plt.plot(X, Y)
    plt.grid(True)
    plt.show()

#Define gamma(theta), which gives the x,y-coordinates of the point on ∂omega identified by the angle theta
def gamma(theta,shape):
    x = math.sin(theta)
    y = 1 - math.cos(theta)
    for i in range(2,len(shape)):
        x += (shape[i]/2)*((1/(1-i))*math.sin((1-i)*theta)+(1/(1+i))*math.sin((1+i)*theta))
        y += (shape[i]/2)*((-1/(1+i))*math.cos((1+i)*theta)+(1/(1+i))-(1/(1-i))*math.cos((1-i)*theta)+(1/(1-i)))     
    return (x,y)


#### FIND THE PERIODIC TRAJECTORIES
#### FIND THE PERIODIC TRAJECTORIES


#Define euclidiean norm of a vector in R2
def norme2(v):
    res = math.sqrt(v[0]**2 + v[1]**2)
    return res

#Define function -L : -(length of the trajectory passing by the points ksi[i])
def minus_L(theta_vector,shape):
    res = 0
    for i in range(len(theta_vector)):
        res += norme2(np.subtract(gamma(theta_vector[i],shape),gamma(theta_vector[i-1],shape)))
    return -res

#Optimize function
def L_maximizer(theta_initial,shape):
    #Perform optimization by minimizing minus L
    result = minimize(minus_L,theta_initial, args=(shape,),tol=1e-3)
    #Get the maximum value of L by negating the minimum value of minus_L
    max_value = -result.fun
    max_point = result.x
    #print(f"Maximum value: {max_value} at point {max_point}")
    return np.mod(max_point, 2 * np.pi)

#Function to obtain ksi from phi
def find_impacts(theta_vector,shape):
    impacts = [gamma(theta_i,shape) for theta_i in theta_vector]
    return impacts

#Print the trajectory of the maximal marked orbit 1/q periodic
def trajectories_visualisation(impacts,shape,q):
    theta = np.linspace(0, 2 * np.pi, 5000)
    x_vals, y_vals = np.array([gamma(t,shape) for t in theta]).T
    impacts.append(impacts[0])
    x_vals1, y_vals1 = np.array(impacts).T
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(x_vals, y_vals, label="Shape", color='blue')
    plt.plot(x_vals1, y_vals1, label="Trajectories", color='red',linestyle='dashed')
    plt.scatter(x_vals1, y_vals1, label="Impacts", color='red', marker='o', s=50,alpha=0.2)
    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.title("Shape in the plan, marked point fixed at (0,0)")
    #plt.title("Maximal marked orbit 1/"+str(q)+"-periodic")
    # Add grid and legend
    plt.legend()
    plt.axis("equal")  # Ensure correct aspect ratio
    plt.grid(True)
    # Save the plot as a PNG image
    #plt.savefig(str(shape)+'_1_'+str(q)+'_periodic_trajectory.png',dpi=300, bbox_inches='tight')
    plt.show()
    del impacts[-1]


#### USE OF LAZUTKIN COORDINATES TO HAVE BETTER INITIAL CONDITIONS WHEN OPTIMIZING L
#### USE OF LAZUTKIN COORDINATES TO HAVE BETTER INITIAL CONDITIONS WHEN OPTIMIZING L


#Define rho to the power 1/3
def rho_function_power_1over3(theta,shape):
    return rho_function(theta,shape)**(1/3)

#Compute C_gamma
def lazutkin_parameter(shape):
    res = integrate.quad(lambda theta: rho_function_power_1over3(theta,shape), 0, 2*math.pi)[0]
    C_shape = (1/res)
    return C_shape

#Define the Lazutkin parametrization x(theta)
def x_lazutkin(theta,shape):
    res = lazutkin_parameter(shape) * integrate.quad(lambda theta_prime: rho_function_power_1over3(theta_prime,shape), 0, theta)[0]
    return res

#Define dx(theta)/dtheta
def d_x_lazutkin(theta,shape):
    res = lazutkin_parameter(shape) * rho_function_power_1over3(theta,shape)
    return res

#Solve x(theta) = y using Newton's method with specified condition initial
def solve_lazutkin_newton(shape,y,initial,):
    x_solution = newton(lambda theta: x_lazutkin(theta, shape) - y, x0=initial, fprime=lambda theta: d_x_lazutkin(theta, shape), maxiter=200)
    return x_solution

#Solve x(theta) = y using root_scalar method
def solve_lazutkin_root(shape,y):
    sol = root_scalar(lambda theta: x_lazutkin(theta,shape) - y, method='brentq', bracket=[0, 2 * math.pi])
    return sol.root

#Find the vector theta0 by solving each component theta0j = j/q with Newton method
def find_theta0_newton(q,shape):
    res = []
    init = math.pi
    for i in range(q):
        y = solve_lazutkin_newton(shape,i/q,init)
        res.append(y)
        init = y
    return res

#Find the vector theta0 by solving each component theta0j = j/q with root_scalar method
def find_theta0_root(q,shape):
    res = []
    for i in range(q):
        y = solve_lazutkin_root(shape,i/q)
        res.append(y)
    return res


#### FIND THE ANGLES IN A TRAJECTORIES
#### FIND THE ANGLES IN A TRAJECTORIES


# Function to calculate the angle of the triangle using the law of cosines
def law_of_cosines(a, b, c):
    if a==0 or b==0:
        print('a ou b est nul')
    return math.acos((a**2 + b**2 - c**2) / (2 * a * b))

# Function to find the angles of the triangle given the coordinates of the three points
def find_angle(x1,x2,x3):
    # Calculate the lengths of the sides
    a = norme2(np.subtract(x1,x2))  # Length of side x1-x2
    b = norme2(np.subtract(x2,x3))  # Length of side x2-x3
    c = norme2(np.subtract(x3,x1))  # Length of side x3-x1
    # Use the law of cosines to find the angle at point x2
    angle = law_of_cosines(a, b, c)
    return angle/2

#Function to obtain phi from ksi
def q_list_phi(impacts):
    res = np.empty(len(impacts),dtype= object)
    for i in range(len(impacts)):
        if i < len(impacts) - 1:
            phi = math.pi/2 - find_angle(impacts[i-1],impacts[i],impacts[i+1])
            res[i] = phi
        else:
            impacts.append(impacts[0])
            phi = math.pi/2 - find_angle(impacts[i-1],impacts[i],impacts[i+1])
            res[i] = phi
            del impacts[-1]
    return res


### CREATION OF T AND T_TILDE
### CREATION OF T AND T_TILDE


def mu_inverse(theta,shape):
    return 2*lazutkin_parameter(shape)*rho_function_power_1over3(theta,shape)

# Ensure phi is a numpy array in T_matrix function and perform element-wise operations
def T_matrix(shape, tronc):
    matrix = np.ones((tronc, tronc))
    matrix[0, :] = mu_inverse(0,shape)
    for i in range(1,tronc):
        theta0 = find_theta0_root(i + 1, shape) # Find theta0 using Root's method
        #theta0 = find_theta0_newton(i+1,shape) # Find theta0 using Newton's method
        #theta0 = [(2*math.pi)*(i/q) for i in range(q)] #Uniform theta0
        theta_impacts = L_maximizer(theta0, shape)
        ksi = np.array([x_lazutkin(theta, shape) for theta in theta_impacts])  # Ensure ksi is a numpy array
        impacts = find_impacts(theta_impacts, shape)
        phi = np.array(q_list_phi(impacts))  # Ensure phi is a numpy array
        #trajectories_visualisation(impacts,shape,i+1)
        for j in range(tronc):
            val = 0
            for s in range(i+1):
                val += math.cos(2*math.pi*(j+1)*ksi[s])*math.sin(phi[s])*mu_inverse(theta_impacts[s],shape)
            matrix[i,j] = val*(1/((i+1)*math.sin(math.pi/(i+1))))
    return matrix

# Function to compute tilde(T)
def tilde(T):
    T_D = np.zeros(np.shape(T))
    for i in range(np.shape(T)[1]):
        for j in range(np.shape(T)[1] - i):
            if j % (i + 1) == 0:
                T_D[i, j + i] = 1/(np.pi)
    inverse_T_D = np.linalg.inv(T_D)
    return np.dot(inverse_T_D, T)

#Plot (q**2)*T(q,j) with j fixed
def plot_mjq(T,column):
    # Select a column index to visualize (ensuring it's within bounds)
    column_index = min(column, np.shape(T)[0] - 1)  # Ensuring the column index does not exceed matrix size
    X = np.array([i for i in range(1,np.shape(T)[0])])
    Y = [q**2*T[q,column_index,] for q in range(1,np.shape(T)[0])]
    plt.figure(figsize=(7, 4))
    plt.plot(X, Y, marker='o', linestyle='-', color='b', label=f'Column {column_index} of T')
    plt.xlabel("Row index")
    plt.ylabel("Matrix values")
    plt.title("Sample of T_matrix")
    plt.legend()
    plt.grid(True)
    plt.show()


### FUNCTIONS TO CALL
### FUNCTIONS TO CALL


def final_spectre(shape,tronc):
    T = T_matrix(shape, tronc)
    T_tilde = tilde(T)
    T_tilde_eigenvalues = np.linalg.eigvals(T_tilde)
    real_parts = np.real(T_tilde_eigenvalues)
    imag_parts = np.imag(T_tilde_eigenvalues)
    plt.figure(figsize=(6, 6))
    plt.scatter(real_parts, imag_parts, color='red', marker='o', label='Eigenvalues',s=50,alpha=0.2)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    end = time.time()
    print(f'Total time with T of size {tronc} is {end - start:.4f} seconds')
    plt.title(f'Eigenvalues in the Complex Plane for '+r'$\tilde{T}$'+f' as a {tronc}*{tronc} matrix. Running time = '+f'{end - start:.4f} seconds'+f' = {(end - start)/60:.4f} minutes')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True)
    plt.legend()
    #plt.savefig(str(omega)+' Eigenvalues_tronc='+str(tronc)+'.png',dpi=300, bbox_inches='tight')
    plt.show()

def final_trajectories(shape,q):
    theta0 = find_theta0_root(q, shape) # Find theta0 using Root's method
    #theta0 = find_theta0_newton(q,shape) # Find theta0 using Newton's method
    #theta0 = [(2*math.pi)*(i/q) for i in range(q)] #Uniform theta0
    theta_impacts = L_maximizer(theta0, shape)
    ksi = np.array([x_lazutkin(theta, shape) for theta in theta_impacts])  # Ensure ksi is a numpy array
    impacts = find_impacts(theta_impacts, shape)
    trajectories_visualisation(impacts,omega,q)


if __name__ == '__main__':
    tronc = 50
    omega = np.array([1])
    #omega = np.array([1 ,0, 0.5, -0.45, 0.3])
    #omega = np.array([1 ,0, 0, 0.99])
    #omega = np.array([1 ,0, 0, -0.999])

    #plot_rho(omega)
    final_trajectories(omega,10)
    #final_spectre(omega,tronc)

    #T = T_matrix(omega, tronc)
    #T_tilde1 = tilde(T)
    #T_tilde = np.transpose(T_tilde1)@T_tilde1

    #T_tilde_eigenvalues = np.linalg.eigvals(T_tilde)
    #real_parts = np.real(T_tilde_eigenvalues)
    #imag_parts = np.imag(T_tilde_eigenvalues)
    #plt.figure(figsize=(6, 6))
    #plt.scatter(real_parts, imag_parts, color='red', marker='o', label='Eigenvalues',s=50,alpha=0.2)
    #plt.xlabel('Real Part')
    #plt.ylabel('Imaginary Part')
    #end = time.time()
    #print(f'Total time with T of size {tronc} is {end - start:.4f} seconds')
    #plt.title(f'Eigenvalues in the Complex Plane for '+r'$\tilde{T}$'+f' as a {tronc}*{tronc} matrix. Running time = '+f'{end - start:.4f} seconds'+f' = {(end - start)/60:.4f} minutes')
    #plt.axhline(0, color='black', linewidth=1)
    #plt.axvline(0, color='black', linewidth=1)
    #plt.grid(True)
    #plt.legend()
    #plt.savefig(str(omega)+' Eigenvalues_tronc='+str(tronc)+'.png',dpi=300, bbox_inches='tight')
    #plt.show()




