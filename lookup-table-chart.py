import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Data from the provided JSON
battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
threat_states = ["Normal", "Confirming", "Confirmed"]
decision_matrix_safe = [[0,0,0],[0,1,0],[1,2,1],[1,2,1],[1,2,1]]
action_labels = ["No DDoS", "XGBoost", "TST"]

# Convert to numpy array for easier handling
matrix = np.array(decision_matrix_safe)

# Create custom colorscale matching the action colors
# 0 = No DDoS = red, 1 = XGBoost = yellow, 2 = TST = green
colorscale = [
    [0.0, '#DB4545'],    # Red for No DDoS (0)
    [0.5, '#D2BA4C'],    # Yellow for XGBoost (1) 
    [1.0, '#2E8B57']     # Green for TST (2)
]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=threat_states,
    y=battery_levels,
    colorscale=colorscale,
    showscale=True,
    colorbar=dict(
        tickmode='array',
        tickvals=[0, 1, 2],
        ticktext=action_labels,
        title="Action"
    ),
    hoverongaps=False,
    hovertemplate='Battery: %{y}<br>Threat: %{x}<br>Action: %{text}<extra></extra>',
    text=[[action_labels[val] for val in row] for row in matrix]
))

# Update layout
fig.update_layout(
    title="Decision Matrix - Safe Temp",
    xaxis_title="Threat State",
    yaxis_title="Battery Level"
)

fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("decision_heatmap.png")