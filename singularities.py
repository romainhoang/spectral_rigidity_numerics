import matplotlib.pyplot as plt
import numpy as np
import math

# Define singularity points
p1 = np.array([-1, 1])
p2 = np.array([1, 1])

# Top arc: circle of radius 1 centered at (0, 1)
center_top = np.array([0, 1])
r_top = 1
theta1_top = np.arctan2(p1[1] - center_top[1], p1[0] - center_top[0])
theta2_top = np.arctan2(p2[1] - center_top[1], p2[0] - center_top[0])
theta_top = np.linspace(theta1_top, theta2_top, 300)
x_top = r_top * np.cos(theta_top) + center_top[0]
y_top = r_top * np.sin(theta_top) + center_top[1]

# Bottom arc: circle centered at (0, 3)
center_bottom = np.array([0, 3])
r_bottom = np.sqrt((1 - center_bottom[0])**2 + (1 - center_bottom[1])**2)
theta1_bottom = np.arctan2(p1[1] - center_bottom[1], p1[0] - center_bottom[0])
theta2_bottom = np.arctan2(p2[1] - center_bottom[1], p2[0] - center_bottom[0])
theta_bottom = np.linspace(theta1_bottom, theta2_bottom, 300)
x_bottom = r_bottom * np.cos(theta_bottom) + center_bottom[0]
y_bottom = r_bottom * np.sin(theta_bottom) + center_bottom[1]

# Top impacts
q1 = 3
theta_top_impacts = np.linspace(theta1_top, theta2_top, q1+2)
x_top_impacts = r_top * np.cos(theta_top_impacts) + center_top[0]
y_top_impacts = r_top * np.sin(theta_top_impacts) + center_top[1]
angle_q1 = math.pi/(2*(q1+1))
print('angle q_1 is')
print(angle_q1)
# Bottom impacts
q2 = 3
theta_bottom_impacts = np.linspace(theta1_bottom, theta2_bottom, q2+2)
x_bottom_impacts = r_bottom * np.cos(theta_bottom_impacts) + center_bottom[0]
y_bottom_impacts = r_bottom * np.sin(theta_bottom_impacts) + center_bottom[1]
angle_q2 = abs(theta1_bottom - theta2_bottom)/(2*(q2+1))
print('angle q_2 is')
print(angle_q2)

# Trajectory
x_trajectory = np.concatenate((x_top_impacts,x_bottom_impacts[::-1]))
y_trajectory = np.concatenate((y_top_impacts,y_bottom_impacts[::-1]))
trajectory = list(zip(x_trajectory,y_trajectory))
#print(trajectory)


# Calculating the tangent vector of the circle arc at point theta
def tangent_vector(theta):
    tangent = np.array([-np.sin(theta), np.cos(theta)])
    return tangent / np.linalg.norm(tangent)

def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    # Clamp value to avoid numerical issues with arccos
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    return np.arccos(cos_theta)

#Function to obtain the angles phi
def q_list_phi_right(impacts,theta_top_imp,theta_bottom_imp):
    theta_bottom_imp_inver = theta_bottom_imp[::-1]
    res = []
    for i in range(len(theta_top_imp)-1):
        v = tangent_vector(theta_top_imp[i])
        diff = (impacts[i+1][0]-impacts[i][0],impacts[i+1][1]-impacts[i][1])
        res.append(angle_between(-v,diff))
    for i in range(len(theta_bottom_imp_inver)-1):
        v = tangent_vector(theta_bottom_imp_inver[i])
        diff = (impacts[len(theta_top_imp)+i+1][0]-impacts[len(theta_top_imp)+i][0],impacts[len(theta_top_imp)+i+1][1]-impacts[len(theta_top_imp)+i][1])
        res.append(angle_between(-v,diff))
    return res

def q_list_phi_left(impacts,theta_top_imp,theta_bottom_imp):
    theta_bottom_imp_inver = theta_bottom_imp[::-1]
    res = []
    v = tangent_vector(theta_bottom_imp_inver[-1])
    diff = (impacts[-2][0]-impacts[-1][0],impacts[-2][1]-impacts[-1][1])
    res.append(angle_between(v,diff))
    for i in range(1,len(theta_top_imp)):
        v = tangent_vector(theta_top_imp[i])
        diff = (impacts[i-1][0]-impacts[i][0],impacts[i-1][1]-impacts[i][1])
        res.append(angle_between(v,diff))
    for i in range(1,len(theta_bottom_imp)):
        v = tangent_vector(theta_bottom_imp_inver[i])
        diff = (impacts[len(theta_top_imp)+i-1][0]-impacts[len(theta_top_imp)+i][0],impacts[len(theta_top_imp)+i-1][1]-impacts[len(theta_top_imp)+i][1])
        res.append(angle_between(v,diff))
    return res

def l_q(theta_top_imp,theta_bottom_imp,traject,q_1,q_2):
    theta_bottom_imp_inver = theta_bottom_imp[::-1]
    phi_right = q_list_phi_right(traject,theta_top_imp,theta_bottom_imp)
    s = 0
    for i in range(q_1+2):
        s += theta_top_imp[i]*math.sin(phi_right[i])
    for i in range(q_2+2):
        s+= theta_bottom_imp_inver[i]*math.sin(q1 + phi_right[i])
    print('l_q value (test) is')
    return s

def is_periodic(theta_top_imp,theta_bottom_imp,traject):
    phi_right = q_list_phi_right(traject,theta_top_imp,theta_bottom_imp)
    phi_left = q_list_phi_left(traject,theta_top_imp,theta_bottom_imp)
    phi_couples = [ (phi_left[i],phi_right[i]) for i in range(len(phi_right))]
    print(phi_couples)
    i = 0
    while i<len(phi_couples):
        if phi_couples[i][0] != phi_couples[i][1]:
            return False
    i = i + 1
    return True

#print(l_q(theta_top_impacts,theta_bottom_impacts,trajectory,q1,q2))
print(is_periodic(theta_top_impacts,theta_bottom_impacts,trajectory))
#u = np.array([1,0])
#v = np.array([1,1])
#print('angle between u and v')
#print(angle_between(-u,v))

# Plotting with circle centers shown
plt.figure(figsize=(6, 6))
plt.plot(x_top, y_top, label="Top Arc (center at (0,1), radius=1)")
plt.plot(x_bottom, y_bottom, label="Bottom Arc (radius=√5 ≈ 2.236)", color='orange')
plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', label="Singularities", zorder=5)

# Marking the centers
#plt.scatter(*center_top, color='blue', s=50, zorder=6, label="Center (0,1)")
#plt.scatter(*center_bottom, color='green', s=50, zorder=6, label="Center (0,3)")

# Annotating the centers
#plt.text(center_top[0] + 0.1, center_top[1], "(0,1)", color='blue', fontsize=9, va='center')
#plt.text(center_bottom[0] + 0.1, center_bottom[1], "(0,3)", color='green', fontsize=9, va='center')

# Annotating the radius
#midpoint = (p1 + p2) / 2
#plt.text(midpoint[0], midpoint[1] - 1.4, "radius = √5 ≈ 2.236", ha='center', fontsize=10, color='orange')

# Plotting trajectories
#plt.plot(x_top_impacts, y_top_impacts, label="Top Trajectories", color='blue',linestyle='dashed')
plt.scatter(x_top_impacts, y_top_impacts, label="Top Impacts", color='blue', marker='o', s=50,alpha=0.2)
#plt.plot(x_bottom_impacts, y_bottom_impacts, label="Bottom Trajectories", color='orange',linestyle='dashed')
plt.scatter(x_bottom_impacts, y_bottom_impacts, label="Bottom Impacts", color='orange', marker='o', s=50,alpha=0.2)
plt.plot(np.concatenate((x_trajectory,[x_trajectory[0]])), np.concatenate((y_trajectory,[y_trajectory[0]])), label="Trajectories", color='red',linestyle='dashed')

# Plotting the 
plt.title("Curve with 2 singularities, q = "+r"$q_1$ + $q_2$ = "+str(q1+q2)+", ("+r"$q_1$="+str(q1)+", "+r"$q_2$="+str(q2)+")")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

