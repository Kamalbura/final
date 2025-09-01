import plotly.graph_objects as go
import numpy as np

# Data from the provided JSON
temperature_levels = ["Safe", "Warning", "Critical"]
threat_states = ["Normal", "Confirming", "Confirmed"]
decision_matrix = [[1,1,0],[2,2,0],[1,1,0]]
action_labels = ["No DDoS", "XGBoost", "TST"]

# Define discrete colors for each action
colors = {
    0: "#DB4545",  # No DDoS (red)
    1: "#D2BA4C",  # XGBoost (yellow)
    2: "#2E8B57"   # TST (green)
}

# Create custom colorscale for discrete values
colorscale = [
    [0.0, colors[0]],
    [0.33, colors[0]],
    [0.33, colors[1]],
    [0.66, colors[1]], 
    [0.66, colors[2]],
    [1.0, colors[2]]
]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=decision_matrix,
    x=temperature_levels,
    y=threat_states,
    colorscale=colorscale,
    zmin=0,
    zmax=2,
    showscale=True,
    colorbar=dict(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=action_labels,
        title="Actions"
    ),
    hovertemplate='Temp: %{x}<br>Threat: %{y}<br>Action: %{text}<extra></extra>',
    text=[[action_labels[val] for val in row] for row in decision_matrix]
))

fig.update_layout(
    title="Decision Matrix Heatmap",
    xaxis_title="Temp Level",
    yaxis_title="Threat State"
)

# Save the chart
fig.write_image("decision_heatmap.png")