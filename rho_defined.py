import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize, root_scalar, newton
import matplotlib.animation as animation
import os
import time
from scipy.interpolate import interp1d

# Start the timer
start = time.time()

eps0 = 0.002
eps = eps0
#eps = 0.002
eta = 0.05


def rho_defined(theta):
    if 0 <= theta < math.pi/4 - eta:
        return 1
    elif math.pi/4 - eta <= theta < math.pi/4 + eta:
        y  = math.pi/4
        phi = (y + eta)/(y - eta)
        b = (eps - phi)/(1 - phi)
        a = (1 - b)/(y - eta)
        return a*theta + b 
    elif math.pi/4 + eta <= theta < 3*math.pi/4 - eta:
        return eps
    elif 3*math.pi/4 - eta <= theta < 3*math.pi/4 + eta:
        y = 3*math.pi/4
        phi = (y + eta)/(y - eta)
        b = (eps - 1/phi)/(1-(1/phi))
        a = (1-b)/(y + eta)
        return a*theta + b 
    elif 3*math.pi/4 + eta <= theta < 5*math.pi/4 - eta:
        return 1
    elif 5*math.pi/4 - eta <= theta < 5*math.pi/4 + eta:
        y  = 5*math.pi/4
        phi = (y + eta)/(y - eta)
        b = (eps - phi)/(1-phi)
        a = (1-b)/(y - eta)
        return a*theta + b
    elif 5*math.pi/4 + eta <= theta < 7*math.pi/4 - eta:
        return eps
    elif 7*math.pi/4 - eta <= theta < 7*math.pi/4 + eta:
        y  = 7*math.pi/4
        phi = (y + eta)/(y - eta)
        b = (eps - 1/phi)/(1-(1/phi))
        a = (1-b)/(y + eta)
        return a*theta + b
    elif 7*math.pi/4 + eta <= theta <= 2*math.pi:
        return 1
    else:
        return eps
    
def plot_rho_defined():
    X = np.linspace(0,2*math.pi,10000)
    Y = [rho_defined(theta) for theta in X]
    Y3 = [0 for i in range(len(X))]
    plt.axvline(x=math.pi/4, color='r', linestyle='--', label=r'x = $\pi$/4')
    plt.axvline(x=3*math.pi/4, color='r', linestyle='--', label=r'x = $3\pi$/4')
    plt.axvline(x=5*math.pi/4, color='r', linestyle='--', label=r'x = $5\pi$/4')
    plt.axvline(x=7*math.pi/4, color='r', linestyle='--', label=r'x = $7\pi$/4')
    plt.plot(X, Y, label=r'$\rho(\theta)$')
    plt.plot(X, Y3, label=r'0')
    plt.grid(True)
    plt.title(r'$\rho(\theta)$ with '+r'$\epsilon,\eta$ = '+f'{eps0},{eta}')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\rho(\theta)$')
    plt.legend()
    plt.show()

def plot_fun(f1,f2,theta_vector):
    print('je suis rentré')
    X = np.linspace(0,2*math.pi,10000)
    Y1 = [f1(theta) for theta in X]
    Y2 = [f2(theta) for theta in X]
    # Create the figure with 2 subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.set_title(r"function $f_x$")
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\rho(\theta)\sin(\theta)$")
    ax1.grid(True)
    ax1.plot(X, Y1, label=r"$f_x(\theta)$", color='blue')
    for i in range(len(theta_vector)):
        ax1.axvline(x=theta_vector[i], color='r', linestyle='--', label=r'x = '+str(theta_vector[i]))      
    ax1.legend()

    ax2.set_title(r"function $f_y$")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$\rho(\theta)\cos(\theta$)")
    ax2.grid(True)
    ax2.plot(X, Y2, label=r"$f_y(\theta)$", color='blue')
    for i in range(len(theta_vector)):
        ax2.axvline(x=theta_vector[i], color='r', linestyle='--', label=r'x = '+str(theta_vector[i]))
    ax2.legend()
    plt.show() 

def gamma_defined_2(theta):
    # Define your function
    fx = lambda u: rho_defined(u)*math.cos(u)
    fy = lambda u: rho_defined(u)*math.sin(u)
    # Integration limits
    a = 0
    b = theta
     # Discontinuity points within [0, 2pi]
    discontinuities = [
        math.pi/4 - eta, math.pi/4 + eta,
        3*math.pi/4 - eta, 3*math.pi/4 + eta,
        5*math.pi/4 - eta, 5*math.pi/4 + eta,
        7*math.pi/4 - eta, 7*math.pi/4 + eta
    ]

    # Filter points actually within [a, b]
    split_points = [p for p in discontinuities if a < p < b]

    # Compute definite integral
    x, errorx = quad(fx, a, b, points=split_points)
    y, errory = quad(fy, a, b, points=split_points)
    #print(r"Integral of $\rho$ from 0 to "+"{theta}"+" = {result}") 
    return (x,y)

def gamma_defined(theta):
    # Function to integrate
    fx = lambda u: rho_defined(u) * math.cos(u)
    fy = lambda u: rho_defined(u) * math.sin(u)

    # Define key discontinuity points
    discontinuities = [
        math.pi/4 - eta, math.pi/4 + eta,
        3*math.pi/4 - eta, 3*math.pi/4 + eta,
        5*math.pi/4 - eta, 5*math.pi/4 + eta,
        7*math.pi/4 - eta, 7*math.pi/4 + eta
    ]

    # Include boundaries 0 and theta
    all_points = [0] + [p for p in discontinuities if 0 < p <= theta] + [theta]
    all_points = sorted(set(all_points))

    # Integrate over subintervals
    total_x = 0
    total_y = 0
    for i in range(len(all_points) - 1):
        a = all_points[i]
        b = all_points[i + 1]
        total_x += quad(fx, a, b, limit=500)[0]
        total_y += quad(fy, a, b, limit=500)[0]

    return (total_x, total_y)


def plot_shape():
    theta = np.linspace(0, 2 * np.pi, 5000)
    x_vals, y_vals = np.array([gamma_defined(t) for t in theta]).T
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(x_vals, y_vals, label="Shape", color='blue')
    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Desired shape")
    # Add grid and legend
    plt.legend()
    plt.axis("equal")  # Ensure correct aspect ratio
    plt.grid(True)
    plt.show()

def f_x(theta):
    y = rho_defined(theta)
    res = y*math.cos(theta)
    return res

def f_y(theta):
    y = rho_defined(theta)
    res = y*math.sin(theta)
    return res

#Define euclidiean norm of a vector in R2
def norme2(v):
    res = math.sqrt(v[0]**2 + v[1]**2)
    return res

#Define function -L : -(length of the trajectory passing by the points ksi[i])
def minus_L(theta_vector):
    res = 0
    for i in range(len(theta_vector)):
        res += norme2(np.subtract(gamma_defined(theta_vector[i]),gamma_defined(theta_vector[i-1])))
    return -res


#Optimize function
def L_maximizer(theta_initial):
    #Perform optimization by minimizing minus L
    result = minimize(minus_L,theta_initial,tol=1e-3)
    #Get the maximum value of L by negating the minimum value of minus_L
    max_value = -result.fun
    max_point = result.x
    #print(f"Maximum value: {max_value} at point {max_point}")
    return np.mod(max_point, 2 * np.pi)

#Define rho to the power 1/3
def rho_function_power_1over3(theta):
    return rho_defined(theta)**(1/3)

#Compute C_gamma
def lazutkin_parameter():
    # Function to integrate
    f = lambda u: rho_function_power_1over3(u)

    # Define key discontinuity points
    discontinuities = [
        math.pi/4 - eta, math.pi/4 + eta,
        3*math.pi/4 - eta, 3*math.pi/4 + eta,
        5*math.pi/4 - eta, 5*math.pi/4 + eta,
        7*math.pi/4 - eta, 7*math.pi/4 + eta
    ]

    # Include boundaries 0 and theta
    all_points = [0] + [p for p in discontinuities] + [2*math.pi]
    all_points = sorted(all_points)

    # Integrate over subintervals
    total = 0
    for i in range(len(all_points) - 1):
        a = all_points[i]
        b = all_points[i + 1]
        total += quad(f, a, b, limit=500)[0]
    res = (1/total)
    return res

C_omega = lazutkin_parameter()

#Define the Lazutkin parametrization x(theta)
def x_lazutkin(theta):
    # Function to integrate
    f = lambda u: rho_function_power_1over3(u)
    # Define key discontinuity points
    discontinuities = [
        math.pi/4 - eta, math.pi/4 + eta,
        3*math.pi/4 - eta, 3*math.pi/4 + eta,
        5*math.pi/4 - eta, 5*math.pi/4 + eta,
        7*math.pi/4 - eta, 7*math.pi/4 + eta
    ]

    # Include boundaries 0 and theta
    all_points = [0] + [p for p in discontinuities if 0 < p <= theta] + [theta]
    all_points = sorted(all_points)

    # Integrate over subintervals
    total = 0
    for i in range(len(all_points) - 1):
        a = all_points[i]
        b = all_points[i + 1]
        total += quad(f, a, b, limit=500)[0]
    res = C_omega * total
    return res


def plot_lazutkin():
    X = np.linspace(0,2*math.pi,10000)
    Y = [x_lazutkin(theta) for theta in X]
    plt.axvline(x=math.pi/4, color='r', linestyle='--', label=r'x = $\pi$/4')
    plt.axvline(x=3*math.pi/4, color='r', linestyle='--', label=r'x = $3\pi$/4')
    plt.axvline(x=5*math.pi/4, color='r', linestyle='--', label=r'x = $5\pi$/4')
    plt.axvline(x=7*math.pi/4, color='r', linestyle='--', label=r'x = $7\pi$/4')
    plt.plot(X, Y, label=r'$\rho(\theta)$')
    plt.grid(True)
    plt.title(r'Lazutkin coordinates $x(\theta)$')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$x(\theta)$')
    plt.legend()
    plt.show()

#Define dx(theta)/dtheta
def d_x_lazutkin(theta):
    res = C_omega * rho_function_power_1over3(theta)
    return res

# Generate a dense grid of theta values
theta_grid = np.linspace(0, 2 * np.pi, 100000)
# Compute x(theta) over the grid
x_grid = np.array([x_lazutkin(t) for t in theta_grid])
# Create interpolator to estimate theta from x
inverse_lazutkin = interp1d(x_grid, theta_grid, bounds_error=False, fill_value="extrapolate")

def solve_lazutkin_interpolation(y):
    return float(inverse_lazutkin(y))

def find_theta0_interpolation(q):
    res = []
    for i in range(q):
        y = solve_lazutkin_interpolation(i/q)
        res.append(y)
    return res

#Solve x(theta) = y using Newton's method with specified condition initial
def solve_lazutkin_newton(y,initial,):
    #print(f"Solving x_lazutkin(theta) = {y} from initial = {initial}")
    x_solution = newton(lambda theta: x_lazutkin(theta) - y, x0=initial, fprime=lambda theta: d_x_lazutkin(theta), maxiter=200)
    return x_solution

#Find the vector theta0 by solving each component theta0j = j/q with Newton method
def find_theta0_newton(q):
    res = []
    init = 0
    for i in range(q):
        y = solve_lazutkin_newton(i/q,init)
        res.append(y)
        init = y
    return res

#Function to obtain ksi from phi
def find_impacts(theta_vector):
    impacts = [gamma_defined(theta_i) for theta_i in theta_vector]
    return impacts

#Print the trajectory of the maximal marked orbit 1/q periodic
def trajectory_visualisation(q):
    start_visu = time.time()
    #theta_init = [(2 * math.pi) * (i / q) for i in range(q)]
    #theta_init = find_theta0_newton(q)
    theta_init = find_theta0_interpolation(q)
    theta_vector = L_maximizer(theta_init)
    impacts = find_impacts(theta_vector)
    theta = np.linspace(0, 2 * np.pi, 5000)
    x_vals, y_vals = np.array([gamma_defined(t) for t in theta]).T
    impacts.append(impacts[0])
    x_vals1, y_vals1 = np.array(impacts).T
    plt.figure(figsize=(7.5, 3.5))
    plt.plot(x_vals, y_vals, label="Shape", color='blue')
    plt.plot(x_vals1, y_vals1, label="Trajectories", color='red',linestyle='dashed')
    plt.scatter(x_vals1, y_vals1, label="Impacts", color='red', marker='o', s=50,alpha=0.2)
    # Add labels and title
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Maximal marked orbit 1/"+str(q)+"-periodic")
    # Add grid and legend
    plt.legend()
    plt.axis("equal")  # Ensure correct aspect ratio
    plt.grid(True)
    # Save the plot as a PNG image
    #plt.savefig(str(shape)+'_1_'+str(q)+'_periodic_trajectory.png',dpi=300, bbox_inches='tight')
    mid_visu = time.time()
    print('Running time for plotting trajectory = '+f'{mid_visu - start_visu:.4f} seconds'+f' = {(mid_visu - start_visu)/60:.4f} minutes'+' (q = '+str(q)+')')
    plt.show()
    del impacts[-1]

def animation_trajectory(q_max):
    start_traj = time.time()
        # Create figure
    fig, ax = plt.subplots()
    ax.set_title(r"Maximal marked orbit 1/"+str(2)+"-periodic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    scatter = ax.scatter([], [], c='b', alpha=0.2,s=50)

    theta = np.linspace(0, 2 * np.pi, 5000)
    x_vals, y_vals = np.array([gamma_defined(t) for t in theta]).T
    curve_line, = ax.plot(x_vals, y_vals, color='blue', label="Shape", lw=2) 

    # Initialize red dashed line for trajectories
    trajectory_line, = ax.plot([], [], 'r--', label="Trajectories", alpha=0.6)

    def update(q):
        # Compute trajectory points
        #theta_init = [(2 * math.pi) * (i / q) for i in range(q)]
        #theta_init = find_theta0_newton(q)
        theta_init = find_theta0_interpolation(q)
        theta_vector = L_maximizer(theta_init)
        trajectories = find_impacts(theta_vector)
        trajectories.append(trajectories[0])  # Close the loop

        x_vals1, y_vals1 = np.array(trajectories).T

        # Update shape (still blue)
        curve_line.set_data(x_vals, y_vals)

        # Update red dotted line
        trajectory_line.set_data(x_vals1, y_vals1)
        trajectory_line.set_color('red')
        trajectory_line.set_linestyle('--')

        # Update red scatter points
        scatter.set_offsets(np.c_[x_vals1, y_vals1])
        scatter.set_color('red')  # Points are now red

        ax.set_title(f"Maximal marked orbit 1/{q}-periodic")
        del trajectories[-1]
            # Autoscale based on new trajectory data
        ax.relim()            # Recalculate limits based on artists
        ax.autoscale_view()   # Apply the new limits


        return scatter, trajectory_line
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=range(2,q_max+1), interval=400,blit=False)
    # Show the animation in a live window
    #plt.show()
    # Make sure the folder exists
    save_folder = '/Users/romainhoang/Desktop/Research S8/Work/Code results'
    os.makedirs(save_folder, exist_ok=True)
    # Define the full save path
    save_path = os.path.join(save_folder, f'trajectories_evolution_defined_up_to_{q_max}.mp4')
    # Save the animation as an MP4 video file
    ani.save(save_path, writer='ffmpeg', fps=8)
    end_traj = time.time()
    print('Running time for creating and saving trajectory animation = '+f'{end_traj - start_traj:.4f} seconds'+f' = {(end_traj - start_traj)/60:.4f} minutes'+' (q_max = '+str(q_max)+')')


def animation_eigenvalues(tronc):
    start_eig = time.time()
    global eps
    # Create a single subplot for eigenvalues
    fig, ax1 = plt.subplots(figsize=(7, 7))

    # Eigenvalues Scatter Plot
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(-1, 1)
    ax1.set_title(r"Eigenvalues of $\tilde{T}$ with eps = {eps0}")
    ax1.set_xlabel("Real Part")
    ax1.set_ylabel("Imaginary Part")
    ax1.grid(True)
    scatter = ax1.scatter([], [], c='b', alpha=0.2, s=50)

    def update(k):
        """Update function for animation"""
        global eps, C_omega
        eps = eps0 * (1 - k / 100)
        C_omega = lazutkin_parameter()

        matrix = T_matrix(tronc)
        matrix_tilde = tilde(matrix)
        eigenvalues = np.linalg.eigvals(matrix_tilde)
        scatter.set_offsets(np.c_[eigenvalues.real, eigenvalues.imag])
        ax1.set_title(r"Eigenvalues of $\tilde{T}$ with $\epsilon$ = "+str(eps)+" (Tronc = "+str(tronc)+")")
        return scatter,
    mid_eig = time.time()
    ax1.text(0.5, 1.05, f'Running time before saving the eigenvalues eps-animation = '+f'{mid_eig - start_eig:.4f} seconds'+f' = {(mid_eig - start_eig)/60:.4f} minutes', ha='center', va='bottom', transform=ax1.transAxes, fontsize=12, color='gray')

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=range(0, 100), interval=100)

    # Save animation
    save_folder = '/Users/romainhoang/Desktop/Research S8/Work/Code results'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, str(eps)+f'_eps_animation_with_tronc={tronc}_eigenvalues_only.mp4')
    ani.save(save_path, writer='ffmpeg', fps=30)

    end_eig = time.time()
    print(f'Running time for creating and saving the eigenvalues eps-animation is {end_eig - start_eig:.4f} seconds = {(end_eig - start_eig)/60:.4f} minutes'+' (tronc = '+str(tronc)+')')
    eps = eps0
    #plt.show()


def plot_theta_impacts(impacts):
    X = np.linspace(0,2*math.pi,10000)
    Y = [rho_defined(theta) for theta in X]
    for i in range(len(impacts)):
        plt.axvline(x=impacts[i], color='r', linestyle='--', label=r'impacts['+str(i)+'] = '+str(impacts[i]))         
    plt.plot(X, Y, label=r'$\rho(\theta)$')
    plt.grid(True)
    plt.title(r'$\rho(\theta)$ in rectangular form')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\rho(\theta)$')
    plt.legend()
    plt.show()


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


def mu_inverse(theta):
    return 2*C_omega*rho_function_power_1over3(theta)

# Ensure phi is a numpy array in T_matrix function and perform element-wise operations
def T_matrix(tronc):
    matrix = np.ones((tronc, tronc))
    matrix[0, :] = mu_inverse(0)
    for i in range(1,tronc):
        theta0 = find_theta0_interpolation(i + 1) # Find theta0 using Root's method
        #theta0 = find_theta0_newton(i+1,shape) # Find theta0 using Newton's method
        #theta0 = [(2*math.pi)*(i/q) for i in range(q)] #Uniform theta0
        theta_impacts = L_maximizer(theta0)
        ksi = np.array([x_lazutkin(theta) for theta in theta_impacts])  # Ensure ksi is a numpy array
        impacts = find_impacts(theta_impacts)
        phi = np.array(q_list_phi(impacts))  # Ensure phi is a numpy array
        #trajectories_visualisation(impacts,shape,i+1)
        for j in range(tronc):
            val = 0
            for s in range(i+1):
                val += math.cos(2*math.pi*(j+1)*ksi[s])*math.sin(phi[s])*mu_inverse(theta_impacts[s])
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

def final_spectre(tronc):
    start_spec = time.time()
    T = T_matrix(tronc)
    T_tilde = tilde(T)
    T_tilde_eigenvalues = np.linalg.eigvals(T_tilde)
    real_parts = np.real(T_tilde_eigenvalues)
    imag_parts = np.imag(T_tilde_eigenvalues)
    plt.figure(figsize=(6, 6))
    plt.scatter(real_parts, imag_parts, color='red', marker='o', label='Eigenvalues',s=50,alpha=0.2)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    end_spec = time.time()
    print(f'Running time for plotting spectrum is {end_spec - start_spec:.4f} seconds'+' (tronc = '+str(tronc)+')')
    plt.title(f'Eigenvalues in the Complex Plane for '+r'$\tilde{T}$'+f' as a {tronc}*{tronc} matrix. Running time = '+f'{end_spec - start_spec:.4f} seconds'+f' = {(end_spec - start_spec)/60:.4f} minutes')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True)
    plt.legend()
    #plt.savefig(str(omega)+' Eigenvalues_tronc='+str(tronc)+'.png',dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    tronc = 10
    q = 11
    q_max = 100
    plot_rho_defined()
    plot_lazutkin()
    trajectory_visualisation(q)
    final_spectre(tronc)
    animation_trajectory(q_max)
    animation_eigenvalues(tronc)
    end = time.time()
    print('Total running time of the code :'+f'{end - start:.4f} seconds = {(end - start)/60:.4f} minutes')

