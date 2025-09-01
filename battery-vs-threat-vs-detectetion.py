import plotly.graph_objects as go
import numpy as np

# Data from JSON
battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
threat_states = ["Normal", "Confirming", "Confirmed"]
decision_matrix = [[0,0,0],[1,2,1],[1,2,1],[1,2,1],[1,2,1]]
action_labels = ["No DDoS", "XGBoost", "TST"]

# Create discrete colorscale: red/yellow/green for 0/1/2
colorscale = [
    [0.0, '#DB4545'],    # Red for "No DDoS" (0)
    [0.33, '#DB4545'],   # Red
    [0.34, '#D2BA4C'],   # Yellow for "XGBoost" (1)  
    [0.66, '#D2BA4C'],   # Yellow
    [0.67, '#2E8B57'],   # Green for "TST" (2)
    [1.0, '#2E8B57']     # Green
]

# Abbreviate labels to meet 15 char limit
battery_abbrev = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
threat_abbrev = ["Normal", "Confirm", "Confirmed"]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=decision_matrix,
    x=threat_abbrev,
    y=battery_abbrev,
    colorscale=colorscale,
    showscale=True,
    zmin=0,
    zmax=2,
    hoverongaps=False,
    hovertemplate='Battery: %{y}<br>Threat: %{x}<br>Action: %{customdata}<extra></extra>',
    customdata=[[action_labels[val] for val in row] for row in decision_matrix],
    colorbar=dict(
        tickvals=[0, 1, 2],
        ticktext=["No DDoS", "XGBoost", "TST"]
    )
))

# Update layout
fig.update_layout(
    title="Battery vs Threat Matrix",
    xaxis_title="Threat State",
    yaxis_title="Battery Level"
)

# Save the chart
fig.write_image("heatmap_decision_matrix.png")
fig.show()