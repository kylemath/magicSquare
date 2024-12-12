import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import gridspec
from scipy.optimize import minimize
import random

# Physics parameters
dt = 0.1
num_points = 16
radius = 5
g = 9.0  # gravity
max_history_points = 100
tilt_damping = 0.99  # damping factor for tilt (less than 1)
critical_tilt = np.pi  # about 60 degrees - point of no return
tilt_velocity = 0  # Add this to track tilt speed

# Declare globals first
global tilt_angle, tilt_direction, angular_velocity, is_fallen
is_fallen = False

# Physics state variables
tilt_angle = 0  # current tilt angle
tilt_direction = 0  # direction of tilt
angular_velocity = 300.0  # initial spin rate (rad/s)
moment_of_inertia = sum(w * radius**2 for w in range(1, num_points + 1))  # I = mr²

# Initialize weights with simple integer values
weights = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 8, 9]  # Integer weights
theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)

# Initialize tracking variables
primary_rotation = 0
force_history_x = []
force_history_y = []
tilt_history = []

def calculate_imbalance(weights, num_points):
    """Calculate static imbalance of the weights"""
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    total_weight = sum(weights)
    
    # Static balance (normalized by total weight)
    com_x = sum(w * np.cos(t) for w, t in zip(weights, theta))
    com_y = sum(w * np.sin(t) for w, t in zip(weights, theta))
    return np.sqrt(com_x**2 + com_y**2) / total_weight

def init_visualization():
    """Initialize the visualization plot"""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    ax3d = fig.add_subplot(gs[0, 0:], projection='3d')
    ax_force = fig.add_subplot(gs[1, 0])
    ax_weights = fig.add_subplot(gs[1, 1], projection='polar')
    ax_tilt = fig.add_subplot(gs[2, 0:])
    
    ax3d.set_xlim(-6, 6)
    ax3d.set_ylim(-6, 6)
    ax3d.set_zlim(-6, 6)
    
    return fig, (ax3d, ax_force, ax_weights, ax_tilt)

def update(frame):
    global primary_rotation, tilt_angle, tilt_direction, angular_velocity, tilt_velocity, is_fallen
    
    # If already fallen, don't update anything
    if is_fallen:
        return
    
    # Calculate center of mass coordinates first
    com_x = sum(w * np.cos(t + primary_rotation) for w, t in zip(weights, theta))
    com_y = sum(w * np.sin(t + primary_rotation) for w, t in zip(weights, theta))
    
    # Update angular velocity (decreases due to friction)
    angular_velocity *= 0.999
    
    # Calculate tilt dynamics
    imbalance_force = np.sqrt(com_x**2 + com_y**2) / sum(weights)  # Normalize by total weight
    tilt_direction = np.arctan2(com_y, com_x)
    
    # Calculate tilt acceleration based on:
    # 1. Gravity (now 0)
    # 2. Gyroscopic stability fighting tilt
    # 3. Imbalance force
    gyroscopic_stability = angular_velocity * 0.1
    # Only use gravity term when g != 0
    tilt_acceleration = (0 if g == 0 else g * np.sin(tilt_angle))
    # Scale imbalance force effect by inverse of angular velocity
    imbalance_effect = imbalance_force / (1 + gyroscopic_stability)
    
    tilt_acceleration += imbalance_effect
    
    # Update tilt velocity and position
    tilt_velocity = tilt_velocity * tilt_damping + tilt_acceleration * dt
    tilt_angle += tilt_velocity * dt
    
    # Check for falling over
    if tilt_angle > critical_tilt:
        is_fallen = True
        print("Top fell over!")
        # Instead of plt.close(), we'll just stop updating
        return
    
    # Limit tilt angle for visualization
    tilt_angle = np.clip(tilt_angle, 0, critical_tilt)
    
    # Update primary rotation
    primary_rotation += angular_velocity * dt
    
    # Calculate 3D positions accounting for tilt
    def get_tilted_coordinates(x, y, z):
        # Rotate points around the tilt axis
        c = np.cos(tilt_angle)
        s = np.sin(tilt_angle)
        # Rotation matrix around tilt direction
        rot_x = np.cos(tilt_direction)
        rot_y = np.sin(tilt_direction)
        
        # Apply tilt transformation
        new_x = x * (rot_x**2 * (1-c) + c) + y * (rot_x*rot_y*(1-c)) + z * (-rot_y*s)
        new_y = x * (rot_x*rot_y*(1-c)) + y * (rot_y**2 * (1-c) + c) + z * (rot_x*s)
        new_z = x * (rot_y*s) + y * (-rot_x*s) + z * c
        
        return new_x, new_y, new_z

    # Regular physics update
    time = frame * dt
    primary_rotation += angular_velocity * dt
    
    # Calculate force vector
    force_scale = 2
    force_x = com_x * force_scale
    force_y = com_y * force_scale
    force_z = -force_scale
    
    # Store force history
    force_history_x.append(force_x)
    force_history_y.append(force_y)
    if len(force_history_x) > max_history_points:
        force_history_x.pop(0)
        force_history_y.pop(0)
    
    # Store tilt history
    tilt_history.append(np.degrees(tilt_angle))  # Convert to degrees for better readability
    if len(tilt_history) > max_history_points:
        tilt_history.pop(0)
    
    # Clear plots
    ax3d.clear()
    ax_force.clear()
    ax_weights.clear()
    ax_tilt.clear()
    
    # 3D Plot setup
    ax3d.set_xlim(-6, 6)
    ax3d.set_ylim(-6, 6)
    ax3d.set_zlim(-6, 6)
    
    # Plot floor and back wall
    xx, yy = np.meshgrid(np.linspace(-6, 6, 5), np.linspace(-6, 6, 5))
    ax3d.plot_surface(xx, yy, np.full_like(xx, -6), alpha=0.1, color='gray')
    ax3d.plot_surface(np.full_like(yy, 6), yy, xx, alpha=0.1, color='gray')
    
    # Plot ring with tilt
    ring_theta = np.linspace(0, 2*np.pi, 100)
    ring_x = radius * np.cos(ring_theta)
    ring_y = radius * np.sin(ring_theta)
    ring_z = np.zeros_like(ring_theta)
    
    tilted_ring = [get_tilted_coordinates(x, y, z) 
                   for x, y, z in zip(ring_x, ring_y, ring_z)]
    ring_x, ring_y, ring_z = zip(*tilted_ring)
    ax3d.plot(ring_x, ring_y, ring_z, 'gray', alpha=0.3)
    
    # Plot weights with tilt
    for i in range(num_points):
        angle = theta[i] + primary_rotation
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        
        # Apply tilt transformation
        x, y, z = get_tilted_coordinates(x, y, z)
        
        # Make size differences more dramatic
        size = weights[i] * 100  # Simple scaling based on integer weight
        # Use a different colormap for better contrast
        color = plt.cm.RdYlBu(weights[i] / max(weights))
        ax3d.scatter(x, y, z, s=size, c=[color], alpha=0.8)
        
        # Add weight labels
        ax3d.text(x*1.1, y*1.1, z*1.1, str(weights[i]), 
                 fontsize=8, alpha=0.7)
    
    # Plot force vector and history
    ax3d.quiver(0, 0, 0, force_x, force_y, force_z, color='red', alpha=0.8, arrow_length_ratio=0.2)
    for i, (fx, fy) in enumerate(zip(force_history_x, force_history_y)):
        alpha = i / len(force_history_x)
        ax3d.scatter(fx, fy, -6, c='red', alpha=alpha*0.5, s=20)
    
    # Plot force magnitude history
    time_points = np.linspace(time - len(force_history_x)*dt, time, len(force_history_x))
    force_magnitudes = [np.sqrt(x**2 + y**2) for x, y in zip(force_history_x, force_history_y)]
    ax_force.plot(time_points, force_magnitudes, 'r-')
    ax_force.set_title('Force Magnitude History')
    ax_force.set_xlabel('Time (s)')
    ax_force.set_ylabel('Force')
    ax_force.grid(True)
    
    # Plot weight distribution
    ax_weights.plot(theta, weights, 'b-o')
    ax_weights.fill(theta, weights, alpha=0.25)
    ax_weights.set_title('Weight Distribution')
    ax_weights.grid(True)
    ax_weights.set_rlim(0, num_points + 1)
    ax_weights.set_rticks(range(0, num_points + 1, 2))
    
    # Plot tilt angle history
    time_points = np.linspace(time - len(tilt_history)*dt, time, len(tilt_history))
    ax_tilt.plot(time_points, tilt_history, 'g-')
    ax_tilt.set_title('Tilt Angle History')
    ax_tilt.set_xlabel('Time (s)')
    ax_tilt.set_ylabel('Tilt Angle (degrees)')
    ax_tilt.grid(True)
    ax_tilt.set_ylim(0, 60)  # Assuming max tilt is π/3 (60 degrees)
    
    # Set 3D view angle
    ax3d.view_init(elev=20, azim=45)

# Initialize visualization and create animation
fig, (ax3d, ax_force, ax_weights, ax_tilt) = init_visualization()
anim = animation.FuncAnimation(fig, update, frames=None, 
                             interval=50, blit=False,
                             cache_frame_data=False,
                             repeat=False)  # Add repeat=False

plt.show()