import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Data from the provided JSON
battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
threat_states = ["Normal", "Confirming", "Confirmed"]
corrected_decision_matrix = [[0, 0, 0], [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1]]
action_labels = ["No DDoS", "XGBoost", "TST"]

# Convert to numpy array for easier handling
matrix = np.array(corrected_decision_matrix)

# Create text annotations for each cell
text_matrix = []
for i in range(len(battery_levels)):
    row_text = []
    for j in range(len(threat_states)):
        action_idx = matrix[i, j]
        row_text.append(action_labels[action_idx])
    text_matrix.append(row_text)

# Create custom colorscale using the brand colors
# 0=No DDoS (cyan), 1=XGBoost (red), 2=TST (sea green)
colorscale = [
    [0.0, '#1FB8CD'],      # Strong cyan for No DDoS
    [0.5, '#DB4545'],      # Bright red for XGBoost  
    [1.0, '#2E8B57']       # Sea green for TST
]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=threat_states,
    y=battery_levels,
    text=text_matrix,
    texttemplate="%{text}",
    textfont={"size": 12},
    colorscale=colorscale,
    showscale=False,
    hovertemplate="Battery: %{y}<br>Threat: %{x}<br>Action: %{text}<extra></extra>"
))

# Update layout
fig.update_layout(
    title="Corrected DDoS Decision Matrix",
    xaxis_title="Threat State",
    yaxis_title="Battery Level"
)

# Update axes
fig.update_xaxes(side="bottom")
fig.update_yaxes(autorange="reversed")  # Reverse y-axis so highest battery is at top

# Save the chart
fig.write_image("decision_matrix_heatmap.png")