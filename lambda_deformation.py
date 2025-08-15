#Import the file that calculates the matrix T_tilde
import main

#Import the needed librairies
import numpy as np
import matplotlib.pyplot as plt
from random import seed, randint
import time
import matplotlib.animation as animation
import os

tronc = 20
#omega = np.array([1])
omega = np.array([1 ,0, 0, 0.99])
theta = np.linspace(0, 2 * np.pi, 5000)

# Create the figure with 2 subplots (side by side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Eigenvalues Scatter Plot
ax1.set_xlim(0, 2)
ax1.set_ylim(-1, 1)
ax1.set_title(r"Eigenvalues of $\tilde{T}$ with lambda = 0")
ax1.set_xlabel("Real Part")
ax1.set_ylabel("Imaginary Part")
ax1.grid(True)
scatter = ax1.scatter([], [], c='b', alpha=0.2, s=50)

ax1.set_title("Simple 2D Plane with Curve")
ax1.set_xlabel("Real Part")
ax1.set_ylabel("Imaginary Part")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(0, 3)
ax2.grid(True)

# Initialize the curve
x_vals_c, y_vals_c = np.array([main.gamma(t,[1]) for t in theta]).T
curve_line, = ax2.plot(x_vals_c, y_vals_c, color='blue', label="Shape", lw=2) 


def update(k):
    """Update function for animation"""
    c = k/100
    temp_tail = c*omega[2:]
    temp_shape = np.concatenate(([1,0], temp_tail))
    x_vals, y_vals = np.array([main.gamma(t,temp_shape) for t in theta]).T
    curve_line.set_data(x_vals, y_vals)  # Show up to `k`-th point of the curve

    matrix = main.T_matrix(temp_shape,tronc)
    matrix_tilde = main.tilde(matrix)
    eigenvalues = np.linalg.eigvals(matrix_tilde)  # Compute eigenvalues
    scatter.set_offsets(np.c_[eigenvalues.real, eigenvalues.imag])  # Update scatter plot
    ax1.set_title(r"Eigenvalues of $\tilde{T}$ with $\lambda$ = "+str(k/100)+" (Tronc = "+str(tronc)+")")
    ax2.set_title(r"$\Omega_{\lambda}$ ($\lambda$ = "+str(k/100)+") with $\Omega_{1}$ is "+str(omega))
    return scatter,

mid = time.time()
ax1.text(0.5, 1.05, f'Running time before saving simulation = '+f'{mid - main.start:.4f} seconds'+f' = {(mid - main.start)/60:.4f} minutes', ha='center', va='bottom', transform=ax1.transAxes, fontsize=12, color='gray')
# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(0, 101), interval=100)
# Make sure the folder exists
#save_folder = '/Users/..' #to be completed
#os.makedirs(save_folder, exist_ok=True)
# Define the full save path
#save_path = os.path.join(save_folder, str(omega)+f'_lambda_deformation_with_tronc={tronc}.mp4')
# Save the animation as an MP4 video file
#ani.save(save_path, writer='ffmpeg', fps=30)

end = time.time()
print(f'Total time for the eigenvalues lambda-animation with tronc = {tronc} is {end - main.start:.4f} seconds'+f' = {(end - main.start)/60:.4f} minutes')
plt.show()
