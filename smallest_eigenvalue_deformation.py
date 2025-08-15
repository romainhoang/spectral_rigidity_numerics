#Import the file that calculates the matrix T_tilde
import main
import close_to_0

#Import the needed librairies
import numpy as np
import matplotlib.pyplot as plt
from random import seed, randint
import time
import matplotlib.animation as animation
import os

close_to_0.eps = 0.1

omega = np.array([1,0,0.5,-0.45,0.3])
#omega = np.array([1,0,0,0.99])
#omega = np.array([1])
tronc = 50

T = main.T_matrix(omega, tronc)
T_tilde = main.tilde(T)
theta = np.linspace(0, 2 * np.pi, 500)

# Create figure
fig, ax = plt.subplots()
ax.set_ylim(-0.1, 2.5)  
ax.set_xlim(-1.75, 1.75)
ax.set_title(r'$\epsilon$ = '+str(close_to_0.eps)+r" and $\Omega$ = "+str(omega)+", with tronc= 0")
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.grid(True)
x_vals, y_vals = np.array([main.gamma(t,omega) for t in theta]).T
initial_line, = ax.plot(x_vals, y_vals, 'b-', lw=1, label="Original Curve")
deformed_line, = ax.plot(x_vals, y_vals, 'r--', lw=1, label="Deformation")
curve_line, = ax.plot(x_vals, y_vals, c='b')

def update(k):
    """Update function for animation"""
    T_tilde_k = T_tilde[:k, :k]  # Extract k × k submatrix
    abs_deformation_values = [abs(close_to_0.eigenvector_deformation(close_to_0.smallest_eigenvalues(T_tilde_k)[1][0],t,omega)) for t in theta]
    max_abs_deformation_values = max(abs_deformation_values)
    if max_abs_deformation_values != 0:
        delta = close_to_0.eps/max_abs_deformation_values
    else:
        print("Deformation is 0 !!!")
        delta = close_to_0.eps
    x_vals1, y_vals1 = np.array([close_to_0.gamma_deformation(t,omega,close_to_0.smallest_eigenvalues(T_tilde_k)[1][0],delta) for t in theta]).T
    deformed_line.set_data(x_vals1, y_vals1)  # Update the plot
    ax.set_title(r'$\epsilon$ = '+str(close_to_0.eps)+r" and $\Omega$ = "+str(omega)+", with tronc= "+str(k))
    return deformed_line,

ax.text(0.5, 1.05, 'Deformation smallest eigenvalue', ha='center', va='bottom', transform=ax.transAxes, fontsize=12, color='gray')
# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(1, T.shape[0]+1), interval=1000)

# Make sure the folder exists
#save_folder = '/Users/..' #to be completed
#os.makedirs(save_folder, exist_ok=True)

# Define the full save path
#save_path = os.path.join(save_folder, str(omega)+'_Smallest_eigenvalue_animation up to'+str(tronc)+'.mp4')

# Save the animation as an MP4 video file
#ani.save(save_path, writer='ffmpeg', fps=30)


end = time.time()
print(f'Total time for the animation of the deformation up to tronc = {tronc} is {end - main.start:.4f} seconds'+f' = {(end - main.start)/60:.4f} minutes')
plt.show()
