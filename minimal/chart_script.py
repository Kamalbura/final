import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create DataFrame from the provided data
data = {
    "episodes": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99],
    "rewards": [46.5,-23.5,-36.5,71.5,30.0,13.5,26.0,25.0,31.5,21.0,-25.5,28.5,24.0,-0.5,60.0,23.0,55.5,23.0,45.5,30.0,28.5,46.5,28.0,35.0,15.5,21.0,40.5,28.5,1.5,18.5,45.0,40.0,38.0,33.0,25.0,21.5,56.0,61.5,20.5,15.5,18.0,51.5,46.5,35.0,1.0,31.0,38.0,11.5,18.0,45.0,46.0,28.5,38.5,25.0,34.5,-20.0,8.0,21.0,20.5,40.5,28.5,34.0,25.0,33.5,36.5,35.0,45.0,38.5,11.0,28.0,33.5,41.5,40.0,23.0,25.5,-11.5,18.5,8.5,28.0,43.0,43.5,40.0,38.5,23.5,28.5,35.5,45.5,40.0,31.0,33.0,15.0,28.0,31.5,15.5,28.5,25.0,45.0,26.0,40.0,28.0],
    "power_consumption": [111.0,72.5,31.0,113.5,100.0,103.0,98.0,85.5,94.0,100.0,42.0,83.0,100.5,72.5,137.5,85.5,126.5,111.0,125.5,108.5,105.5,123.0,108.5,116.0,86.0,80.5,119.5,94.5,58.5,90.0,113.0,115.0,126.0,111.0,103.5,108.0,126.0,143.5,95.0,70.5,72.0,126.5,126.5,114.0,64.0,105.5,116.0,83.0,86.5,121.0,123.5,94.0,116.5,100.0,108.5,38.0,64.0,83.0,95.0,119.5,105.5,111.0,100.0,108.5,116.5,116.0,121.0,116.5,80.0,100.0,111.5,124.5,115.0,100.0,100.5,25.0,86.5,69.0,100.0,121.0,123.5,115.0,116.5,100.5,105.5,113.5,126.5,115.0,105.5,111.0,80.0,100.0,108.5,80.0,105.5,100.0,121.0,94.0,115.0,100.0],
    "detections": [10,6,2,10,8,9,8,7,8,8,3,7,8,6,11,7,10,9,10,9,9,10,9,10,7,6,10,8,4,7,10,10,10,9,8,9,10,12,8,5,6,10,10,10,5,8,10,7,7,10,10,8,10,8,9,3,5,7,8,10,9,9,8,9,10,10,10,10,6,8,9,10,10,8,8,2,7,5,8,10,10,10,10,8,9,10,10,10,8,9,6,8,9,6,9,8,10,8,10,8]
}

df = pd.DataFrame(data)

# Create the figure
fig = go.Figure()

# Add the scatter plot for episode rewards
fig.add_trace(go.Scatter(
    x=df['episodes'],
    y=df['rewards'],
    mode='lines+markers',
    name='Ep Rewards',
    line=dict(color='#1FB8CD', width=2),
    marker=dict(size=5, color='#1FB8CD'),
    hovertemplate='Ep: %{x}<br>Reward: %{y}<extra></extra>'
))

# Calculate trend line using numpy
z = np.polyfit(df['episodes'], df['rewards'], 1)
p = np.poly1d(z)
trend_y = p(df['episodes'])

# Add trend line
fig.add_trace(go.Scatter(
    x=df['episodes'],
    y=trend_y,
    mode='lines',
    name='Trend Line',
    line=dict(color='#DB4545', width=3, dash='dash'),
    hovertemplate='Ep: %{x}<br>Trend: %{y:.1f}<extra></extra>'
))

# Add power consumption as scatter points (scaled to fit reward range)
power_scaled = (df['power_consumption'] - df['power_consumption'].min()) / (df['power_consumption'].max() - df['power_consumption'].min()) * 100 - 50
fig.add_trace(go.Scatter(
    x=df['episodes'],
    y=power_scaled,
    mode='markers',
    name='Power (scaled)',
    marker=dict(size=4, color='#2E8B57', opacity=0.6),
    hovertemplate='Ep: %{x}<br>Power: %{customdata:.1f}<extra></extra>',
    customdata=df['power_consumption']
))

# Add detection success rate (assuming max 12 detections possible)
detection_rate = (df['detections'] / 12) * 100 - 50  # Scale to fit reward range
fig.add_trace(go.Scatter(
    x=df['episodes'],
    y=detection_rate,
    mode='markers',
    name='Detect Rate',
    marker=dict(size=4, color='#5D878F', opacity=0.6, symbol='diamond'),
    hovertemplate='Ep: %{x}<br>Detections: %{customdata}<extra></extra>',
    customdata=df['detections']
))

# Update layout
fig.update_layout(
    title='RL Agent Training Progress',
    xaxis_title='Episode',
    yaxis_title='Reward Value',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image('ql_training_progress.png')