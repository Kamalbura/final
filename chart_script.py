import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Data from the provided JSON
battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
temperature_levels = ["Safe", "Warning", "Critical"]
threat_states = ["Normal", "Confirming", "Confirmed"]
actions = [[0,0,0,1,1], [0,0,0,2,2], [0,0,0,1,1], [0,0,0,2,2], [0,0,0,1,1], [0,0,0,2,2], [0,0,0,1,1], [0,0,0,2,2], [0,0,0,1,1]]
action_labels = ["No DDoS", "XGBoost", "TST"]

# Color mapping for actions
colors = ["#1FB8CD", "#DB4545", "#2E8B57"]  # Using the brand colors
action_colors = {0: colors[0], 1: colors[1], 2: colors[2]}

# Prepare data for 3D visualization
x_vals = []  # Battery levels
y_vals = []  # Temperature levels  
z_vals = []  # Threat states
action_vals = []
hover_text = []

# Create coordinate mappings
battery_map = {level: i for i, level in enumerate(battery_levels)}
temp_map = {level: i for i, level in enumerate(temperature_levels)}
threat_map = {state: i for i, state in enumerate(threat_states)}

# Process the actions data
action_idx = 0
for threat_idx, threat in enumerate(threat_states):
    for temp_idx, temp in enumerate(temperature_levels):
        for battery_idx, battery in enumerate(battery_levels):
            action_val = actions[action_idx][battery_idx]
            
            x_vals.append(battery_idx)
            y_vals.append(temp_idx)
            z_vals.append(threat_idx)
            action_vals.append(action_val)
            
            hover_text.append(f"Battery: {battery}<br>Temp: {temp}<br>Threat: {threat}<br>Action: {action_labels[action_val]}")
        
        action_idx += 1

# Create 3D scatter plot with bars
fig = go.Figure()

# Add traces for each action type
for action_type in range(3):
    mask = [action == action_type for action in action_vals]
    
    fig.add_trace(go.Scatter3d(
        x=[x_vals[i] for i in range(len(x_vals)) if mask[i]],
        y=[y_vals[i] for i in range(len(y_vals)) if mask[i]], 
        z=[z_vals[i] for i in range(len(z_vals)) if mask[i]],
        mode='markers',
        marker=dict(
            size=15,
            color=colors[action_type],
            symbol='cube'
        ),
        name=action_labels[action_type],
        hovertext=[hover_text[i] for i in range(len(hover_text)) if mask[i]],
        hoverinfo='text'
    ))

# Update layout
fig.update_layout(
    title="Algorithm Actions by System State",
    scene=dict(
        xaxis=dict(
            title="Battery Level",
            tickmode='array',
            tickvals=list(range(5)),
            ticktext=[level[:8] for level in battery_levels]  # Abbreviated
        ),
        yaxis=dict(
            title="Temperature",
            tickmode='array', 
            tickvals=list(range(3)),
            ticktext=temperature_levels
        ),
        zaxis=dict(
            title="Threat State", 
            tickmode='array',
            tickvals=list(range(3)),
            ticktext=threat_states
        )
    ),
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Save the chart
fig.write_image("algorithm_actions_3d.png")
fig.show()