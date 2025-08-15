#Import the file that calculates the matrix T_tilde
import main

#Import the needed librairies
import numpy as np
import matplotlib.pyplot as plt
from random import seed, randint
import time
import matplotlib.animation as animation
import os

omega = np.array([1 ,0, 0.5, -0.45, 0.3])
#omega = np.array([1])
tronc = 50

T = main.T_matrix(omega, tronc)
T_tilde = main.tilde(T)

# Create figure
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)  # Adjust limits based on expected eigenvalues
ax.set_ylim(-2, 2)
ax.set_title(r"Eigenvalues of $\tilde{T}_k$ (k=0)")
ax.set_xlabel("Real Part")
ax.set_ylabel("Imaginary Part")
ax.grid(True)
scatter = ax.scatter([], [], c='b', alpha=0.2,s=50)

def update(k):
    """Update function for animation"""
    T_tilde_k = T_tilde[:k, :k]  # Extract k × k submatrix
    eigenvalues = np.linalg.eigvals(T_tilde_k)  # Compute eigenvalues
    scatter.set_offsets(np.c_[eigenvalues.real, eigenvalues.imag])  # Update scatter plot
    ax.set_title(r"Eigenvalues of $\tilde{T}_{k}$ (k="+str(k)+")")
    return scatter,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(1, T.shape[0]+1), interval=300)

# Make sure the folder exists
#save_folder = '/Users/..'
#os.makedirs(save_folder, exist_ok=True)

# Define the full save path
#save_path = os.path.join(save_folder, f'eigenvalues_animation_up_to_{tronc}.mp4')

# Save the animation as an MP4 video file
#ani.save(save_path, writer='ffmpeg', fps=30)


end = time.time()
print(f'Total time for the animation up to {tronc} eigenvalues is {end - main.start:.4f} seconds'+f' = {(end - main.start)/60:.4f} minutes')
plt.show()
