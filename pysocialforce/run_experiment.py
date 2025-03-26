from pathlib import Path
import numpy as np
import pysocialforce as psf
from pysocialforce import forces
import matplotlib.pyplot as plt
from pysocialforce.utils import data_loader

# Load the data
csv_file = '../data/trajectories_with_velocity.csv'  # Update with actual path
initial_state = data_loader(csv_file, start_timestep_id=5, end_timestep_id=45)


# social groups informoation is represented as lists of indices of the state array
groups = [[0, 1], [2], [3]] 

# list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
# obs = [[1, 2, 7, 8]]
obs = None
# initiate the simulator,
s = psf.Simulator(
    initial_state,
    groups=groups,
    obstacles=obs,
    config_file=Path(__file__).resolve().parent.joinpath("custom_config.toml"),
)
s.step(20)


# Launch animation
with psf.plot.SceneVisualizer(s, "../results/res") as sv:
    sv.animate()
    sv.plot()


des_force = forces.DesiredForce()
des_force.init(s, s.config)
des_force.factor = 1.0

print("The force", des_force.get_force())
print(s, s.config)


social_force = forces.SocialForce()
goal_attractive_force = forces.GoalAttractiveForce()

# Initialize the forces and calculate their force vectors
force_vectors = {}

# Initialize and get force vectors
social_force.init(s, s.config)
goal_attractive_force.init(s, s.config)

# Store force vectors in the dictionary
force_vectors['SocialForce'] = social_force.get_force()
force_vectors['GoalAttractiveForce'] = goal_attractive_force.get_force()

# Extract positions (assuming initial_state is defined)
px = initial_state[:, 0]
py = initial_state[:, 1]

# Plot vectors for SocialForce and GoalAttractiveForce
plt.figure(figsize=(12, 12))


# Extract force vectors and plot them
for force_name, force_array in force_vectors.items():
    fx, fy = force_array[:, 0], force_array[:, 1]

    # Set color based on force name
    color = 'purple' if force_name == 'SocialForce' else 'blue'

    # Plot force vectors with specified color
    plt.quiver(px, py, fx, fy, angles="xy", scale_units="xy", scale=1, color=color, label=force_name)

# Plot initial positions
plt.scatter(px, py, color="b", label="Initial Position", s=50)

# Set axis limits with some padding
x_min, x_max = px.min() - 50, px.max() + 50
y_min, y_max = py.min() - 50, py.max() + 50
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add labels and legend
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.title("Pedestrian Forces and Initial Positions")
plt.grid(True)
plt.axis("equal")  
plt.show()


