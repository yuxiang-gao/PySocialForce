import pandas as pd
import numpy as np

def load_initial_state(csv_file, start_timestep_id: int, end_timestep_id: int) -> np.array:
    """
    Loads pedestrian tracking data and extracts the initial state in the form:
    (Track_ID, px, py, vx, vy, gx, gy), using:
      - The **third occurrence** as the initial state.
      - The **tenth occurrence** as the goal position (or last position if not available).
    
    Parameters:
        csv_file (str): Path to the CSV file containing pedestrian trajectory data.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Track_ID', 'px', 'py', 'vx', 'vy', 'gx', 'gy']
    """
    # Load the dataset
    df = pd.read_csv(csv_file)
    df[["X", "Y", "Vx", "Vy"]] = df[["X", "Y", "Vx", "Vy"]].astype(float)

    # Group data by Track_ID (each pedestrian)
    grouped = df.groupby("Track_ID")

    # Create initial_state list
    initial_state = []

    for track_id, group in grouped:
        if len(group) >= start_timestep_id:
            start_timestep_id = group.iloc[start_timestep_id]  
        else:
            start_timestep_id = group.iloc[-1]  

        if len(group) >= end_timestep_id:
            end_timestep_id = group.iloc[end_timestep_id]  
        else:
            end_timestep_id = group.iloc[-1]  

        # Extract position, velocity, and goal
        px, py = start_timestep_id["X"], start_timestep_id["Y"]
        vx, vy = start_timestep_id["Vx"], start_timestep_id["Vy"]
        gx, gy = end_timestep_id["X"], end_timestep_id["Y"]  

        initial_state.append([px, py, vx, vy, gx, gy])

    return np.array(initial_state)

# Example usage
csv_file = '../data/trajectories_with_velocity.csv'  
initial_state_df = load_initial_state(csv_file)

# Display shape & first few rows
print(f"Shape: {initial_state_df.shape}")
print(initial_state_df.head())  
