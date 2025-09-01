import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import json
import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import UAVDDoSLogger

# Initialize app
app = dash.Dash(__name__, title="UAV DDoS-RL Hybrid Monitor")

# Define color schemes for thermal zones
thermal_colors = {
    'cold': '#00ffff',   # Cyan
    'cool': '#00ff00',   # Green
    'warm': '#ffff00',   # Yellow
    'hot': '#ff9900',    # Orange
    'danger': '#ff0000'  # Red
}

# App layout - Enhanced for hybrid RL monitoring
app.layout = html.Div([
    html.H1("UAV DDoS-RL Hybrid Agent Monitoring Dashboard"),
    
    # Top status row
    html.Div([
        # System status panel
        html.Div([
            html.H3("System Status"),
            html.Div(id='system-status'),
            dcc.Interval(id='status-interval', interval=5000)  # 5 seconds
        ], className='status-panel', style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Thermal status panel - NEW
        html.Div([
            html.H3("Thermal Status", style={'color': '#ff6600'}),
            html.Div(id='thermal-status'),
            dcc.Interval(id='thermal-interval', interval=3000)  # 3 seconds - more frequent for critical metrics
        ], className='thermal-panel', style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '5px'}),
        
        # Expert agreement panel - NEW
        html.Div([
            html.H3("RL Decision Sources", style={'color': '#0066cc'}),
            html.Div(id='expert-agreement'),
            dcc.Interval(id='expert-interval', interval=5000)  # 5 seconds
        ], className='expert-panel', style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'backgroundColor': '#f9f9f9', 'padding': '10px', 'borderRadius': '5px'})
    ], className='top-row', style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Middle row with thermal and decision graphs
    html.Div([
        # Temperature monitoring graph - NEW
        html.Div([
            html.H3("Temperature Monitoring"),
            dcc.Graph(id='temperature-graph'),
            dcc.Interval(id='temp-interval', interval=10000)  # 10 seconds
        ], className='graph-panel', style={'width': '48%', 'display': 'inline-block'}),
        
        # Decision history with source tracking
        html.Div([
            html.H3("Decision History"),
            dcc.Graph(id='decision-history'),
            dcc.Interval(id='history-interval', interval=10000)  # 10 seconds
        ], className='graph-panel', style={'width': '48%', 'display': 'inline-block'})
    ], className='middle-row', style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Bottom row with power and safety metrics
    html.Div([
        html.Div([
            html.H3("Power Consumption"),
            dcc.Graph(id='power-consumption'),
            dcc.Interval(id='power-interval', interval=10000)  # 10 seconds
        ], className='graph-panel', style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Expert vs Neural Network Usage"),
            dcc.Graph(id='expert-vs-neural'),
            dcc.Interval(id='expert-neural-interval', interval=15000)  # 15 seconds
        ], className='graph-panel', style={'width': '48%', 'display': 'inline-block'})
    ], className='bottom-row', style={'display': 'flex', 'justifyContent': 'space-between'}),
    
    # Discretized state visualization - NEW
    html.Div([
        html.H3("Current State Discretization"),
        html.Div(id='discretized-state'),
        dcc.Interval(id='state-interval', interval=5000)  # 5 seconds
    ], className='state-panel', style={'marginTop': '20px', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#f0f0f0'}),
    
    # Hidden div for storing data
    html.Div(id='decision-data', style={'display': 'none'}),
    
    # Update interval
    dcc.Interval(id='update-interval', interval=5000)  # 5 seconds
])

# Callback to load decision data
@app.callback(
    Output('decision-data', 'children'),
    Input('update-interval', 'n_intervals')
)
def update_decision_data(n):
    # In production, this would load from a real-time source
    # For demo, we load from a file
    try:
        with open(os.path.join('logs', 'decision_log.json'), 'r') as f:
            return json.dumps(json.load(f))
    except Exception as e:
        return json.dumps([])

# Callback to update system status
@app.callback(
    Output('system-status', 'children'),
    Input('decision-data', 'children')
)
def update_system_status(data):
    if not data:
        return html.Div("No data available", style={'color': 'red'})
    
    decisions = json.loads(data)
    if not decisions:
        return html.Div("No decisions recorded", style={'color': 'orange'})
    
    # Get latest decision
    latest = decisions[-1]
    
    # Calculate summary stats
    total = len(decisions)
    expert_matches = sum(d.get('expert_alignment', False) for d in decisions)
    violations = sum(1 for d in decisions if d.get('safety_violation', False))
    
    status_style = {'color': 'green'} if not violations else {'color': 'red'}
    
    # Enhanced status display
    return html.Div([
        html.Div(f"Latest Decision: {latest.get('action_label', 'Unknown')} at {latest.get('timestamp', '')[:19]}", 
                style={'fontWeight': 'bold'}),
        html.Div([
            html.Span("Battery: ", style={'fontWeight': 'bold'}),
            html.Span(latest.get('state', {}).get('battery', 'Unknown'))
        ]),
        html.Div([
            html.Span("Threat: ", style={'fontWeight': 'bold'}),
            html.Span(latest.get('state', {}).get('threat', 'Unknown'))
        ]),
        html.Div([
            html.Span("Safety Status: ", style={'fontWeight': 'bold'}),
            html.Span("GOOD" if not violations else f"VIOLATIONS: {violations}", style=status_style)
        ]),
        html.Div(f"Expert Alignment: {expert_matches/total:.1%} ({expert_matches}/{total})"),
        html.Div(f"Algorithm Executions: {total}")
    ])

# NEW: Callback for thermal status
@app.callback(
    Output('thermal-status', 'children'),
    Input('decision-data', 'children')
)
def update_thermal_status(data):
    if not data:
        return html.Div("No thermal data", style={'color': 'gray'})
    
    decisions = json.loads(data)
    if not decisions:
        return html.Div("No thermal records", style={'color': 'gray'})
    
    # Get latest decision with temperature info
    latest = decisions[-1]
    
    # Extract temperature and determine zone
    temp = latest.get('state', {}).get('temperature', 'Unknown')
    
    # Determine temperature zone and value
    if isinstance(temp, str):
        # Handle string temperature categories
        if temp == "Safe":
            temp_zone = "cool"
            temp_value = 50
        elif temp == "Warning":
            temp_zone = "warm" 
            temp_value = 65
        elif temp == "Critical":
            temp_zone = "danger"
            temp_value = 80
        else:
            try:
                # Try to parse as a numeric string
                temp_value = float(temp)
                if temp_value < 55:
                    temp_zone = "cold"
                elif temp_value < 60:
                    temp_zone = "cool"
                elif temp_value < 65:
                    temp_zone = "warm"
                elif temp_value < 70:
                    temp_zone = "hot"
                else:
                    temp_zone = "danger"
            except:
                temp_zone = "unknown"
                temp_value = 0
    else:
        # Handle numeric temperature
        try:
            temp_value = float(temp)
            if temp_value < 55:
                temp_zone = "cold"
            elif temp_value < 60:
                temp_zone = "cool"
            elif temp_value < 65:
                temp_zone = "warm"
            elif temp_value < 70:
                temp_zone = "hot"
            else:
                temp_zone = "danger"
        except:
            temp_zone = "unknown"
            temp_value = 0
    
    # Get color for the temperature zone
    zone_color = thermal_colors.get(temp_zone, '#888888')
    
    # Create thermal gauge visualization
    return html.Div([
        html.Div([
            html.Span("Current Temperature: ", style={'fontWeight': 'bold'}),
            html.Span(f"{temp}", style={'color': zone_color, 'fontWeight': 'bold'})
        ]),
        html.Div([
            html.Span("Temperature Zone: ", style={'fontWeight': 'bold'}),
            html.Span(f"{temp_zone.upper()}", style={'color': zone_color, 'fontWeight': 'bold'})
        ]),
        html.Div([
            html.Div(style={
                'backgroundColor': '#e0e0e0',
                'width': '100%',
                'height': '20px',
                'borderRadius': '10px',
                'margin': '10px 0'
            }, children=[
                html.Div(style={
                    'backgroundColor': zone_color,
                    'width': f"{min(100, max(0, temp_value - 30) * 2)}%",
                    'height': '100%',
                    'borderRadius': '10px',
                    'transition': 'width 0.5s ease'
                })
            ])
        ]),
        html.Div([
            html.Span("30°C", style={'float': 'left'}),
            html.Span("50°C", style={'position': 'absolute', 'left': '40%', 'transform': 'translateX(-50%)'}),
            html.Span("70°C", style={'float': 'right'})
        ], style={'position': 'relative', 'width': '100%'}),
        html.Div([
            html.Span("Thermal Management: ", style={'fontWeight': 'bold'}),
            html.Span("ACTIVE" if temp_zone in ["warm", "hot", "danger"] else "PASSIVE", 
                    style={'color': 'red' if temp_zone in ["hot", "danger"] else 'green'})
        ], style={'marginTop': '10px'})
    ])

# NEW: Callback for expert agreement panel
@app.callback(
    Output('expert-agreement', 'children'),
    Input('decision-data', 'children')
)
def update_expert_agreement(data):
    if not data:
        return html.Div("No decision data", style={'color': 'gray'})
    
    decisions = json.loads(data)
    if not decisions:
        return html.Div("No decisions recorded", style={'color': 'gray'})
    
    # Count decision sources (these should be tracked in the decision log)
    # For now we'll simulate with random data if not present
    expert_count = sum(1 for d in decisions if d.get('decision_source') == 'expert_table')
    qtable_count = sum(1 for d in decisions if d.get('decision_source') == 'q_table')
    neural_count = sum(1 for d in decisions if d.get('decision_source') == 'neural_network')
    
    # If no decision sources are recorded, estimate based on expert alignment
    if expert_count + qtable_count + neural_count == 0:
        expert_count = sum(1 for d in decisions if d.get('expert_alignment', False))
        # Simulate the rest with reasonable distribution
        total = len(decisions)
        remaining = total - expert_count
        qtable_count = int(remaining * 0.7)
        neural_count = remaining - qtable_count
    
    total = expert_count + qtable_count + neural_count
    
    # Calculate percentages
    expert_pct = expert_count / total * 100 if total > 0 else 0
    qtable_pct = qtable_count / total * 100 if total > 0 else 0
    neural_pct = neural_count / total * 100 if total > 0 else 0
    
    return html.Div([
        html.Div([
            html.Span("Expert Table: ", style={'fontWeight': 'bold'}),
            html.Span(f"{expert_count} ({expert_pct:.1f}%)", style={'color': '#006600'})
        ]),
        html.Div([
            html.Span("Q-Table: ", style={'fontWeight': 'bold'}),
            html.Span(f"{qtable_count} ({qtable_pct:.1f}%)", style={'color': '#0000cc'})
        ]),
        html.Div([
            html.Span("Neural Network: ", style={'fontWeight': 'bold'}),
            html.Span(f"{neural_count} ({neural_pct:.1f}%)", style={'color': '#cc6600'})
        ]),
        html.Div(style={
            'backgroundColor': '#e0e0e0',
            'width': '100%',
            'height': '20px',
            'borderRadius': '10px',
            'margin': '10px 0',
            'display': 'flex'
        }, children=[
            # Expert table portion
            html.Div(style={
                'backgroundColor': '#006600',
                'width': f"{expert_pct}%",
                'height': '100%',
                'borderTopLeftRadius': '10px',
                'borderBottomLeftRadius': '10px',
            }),
            # Q-table portion
            html.Div(style={
                'backgroundColor': '#0000cc',
                'width': f"{qtable_pct}%",
                'height': '100%',
            }),
            # Neural network portion
            html.Div(style={
                'backgroundColor': '#cc6600',
                'width': f"{neural_pct}%",
                'height': '100%',
                'borderTopRightRadius': '10px',
                'borderBottomRightRadius': '10px',
            })
        ]),
        html.Div([
            html.Div("Expert", style={'float': 'left', 'color': '#006600'}),
            html.Div("Q-Table", style={'textAlign': 'center', 'color': '#0000cc'}),
            html.Div("Neural", style={'float': 'right', 'color': '#cc6600'})
        ], style={'position': 'relative', 'width': '100%'})
    ])

# NEW: Callback for temperature graph
@app.callback(
    Output('temperature-graph', 'figure'),
    Input('decision-data', 'children')
)
def update_temperature_graph(data):
    if not data:
        return go.Figure()
    
    decisions = json.loads(data)
    if not decisions:
        return go.Figure()
    
    df = pd.DataFrame(decisions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract temperature data - handle both string and numeric values
    def extract_temp_value(state_dict):
        temp = state_dict.get('temperature', 0)
        if isinstance(temp, str):
            if temp == "Safe":
                return 50
            elif temp == "Warning":
                return 65
            elif temp == "Critical":
                return 80
            else:
                try:
                    return float(temp.replace('%', ''))
                except:
                    return 30
        return float(temp)
    
    df['temp_value'] = df['state'].apply(extract_temp_value)
    
    # Create figure with temperature plot
    fig = go.Figure()
    
    # Add temperature trace
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temp_value'],
        mode='lines',
        name='Temperature',
        line=dict(color='red', width=2)
    ))
    
    # Add warning threshold
    fig.add_trace(go.Scatter(
        x=[df['timestamp'].min(), df['timestamp'].max()],
        y=[65, 65],
        mode='lines',
        name='Warning Threshold',
        line=dict(color='orange', width=1, dash='dash')
    ))
    
    # Add critical threshold
    fig.add_trace(go.Scatter(
        x=[df['timestamp'].min(), df['timestamp'].max()],
        y=[70, 70],
        mode='lines',
        name='Critical Threshold',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    # Add shaded regions for temperature zones
    fig.add_trace(go.Scatter(
        x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
        y=[70] * len(df) + [100] * len(df),
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Danger Zone'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
        y=[65] * len(df) + [70] * len(df),
        fill='toself',
        fillcolor='rgba(255,165,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Warning Zone'
    ))
    
    # Update layout
    fig.update_layout(
        title='Temperature Monitoring',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        legend_title='',
        yaxis=dict(range=[30, 85])
    )
    
    return fig

# Enhanced callback for decision history graph
@app.callback(
    Output('decision-history', 'figure'),
    Input('decision-data', 'children')
)
def update_decision_history(data):
    if not data:
        return go.Figure()
    
    decisions = json.loads(data)
    if not decisions:
        return go.Figure()
    
    df = pd.DataFrame(decisions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create action counts by time
    action_counts = df.groupby([pd.Grouper(key='timestamp', freq='5min'), 'action_label']).size().unstack(fill_value=0)
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping for algorithms
    colors = {'No_DDoS': 'green', 'XGBoost': 'blue', 'TST': 'red'}
    
    for action in action_counts.columns:
        fig.add_trace(go.Scatter(
            x=action_counts.index,
            y=action_counts[action],
            mode='lines',
            name=action,
            line=dict(width=2, color=colors.get(action, 'gray'))
        ))
    
    # Add markers for thermal events if available
    if 'thermal_event' in df.columns:
        thermal_events = df[df['thermal_event'] == True]
        if len(thermal_events) > 0:
            fig.add_trace(go.Scatter(
                x=thermal_events['timestamp'],
                y=[0] * len(thermal_events),
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='red',
                ),
                name='Thermal Event'
            ))
    
    fig.update_layout(
        title='Decision History by Algorithm',
        xaxis_title='Time',
        yaxis_title='Count',
        legend_title='Algorithm'
    )
    
    return fig

# Enhanced callback for power consumption graph
@app.callback(
    Output('power-consumption', 'figure'),
    Input('decision-data', 'children')
)
def update_power_consumption(data):
    if not data:
        return go.Figure()
    
    decisions = json.loads(data)
    if not decisions:
        return go.Figure()
    
    df = pd.DataFrame(decisions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create cumulative power by time
    df = df.sort_values('timestamp')
    df['cumulative_power'] = df['power_cost'].cumsum()
    
    # Create figure
    fig = go.Figure()
    
    # Add cumulative power trace
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_power'],
        mode='lines',
        name='Cumulative Power',
        line=dict(color='blue', width=2)
    ))
    
    # Add power per decision as bars
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['power_cost'],
        name='Power per Decision',
        marker_color=df['power_cost'].apply(lambda x: 
            'green' if x < 4.5 else
            'orange' if x < 6.0 else
            'red'),
        yaxis='y2'
    ))
    
    # Add power thresholds
    fig.add_trace(go.Scatter(
        x=[df['timestamp'].min(), df['timestamp'].max()],
        y=[6.0, 6.0],
        mode='lines',
        name='Power Warning',
        line=dict(color='red', width=1, dash='dash'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Power Consumption',
        xaxis_title='Time',
        yaxis_title='Cumulative Power (W)',
        yaxis2=dict(
            title='Power per Decision (W)',
            overlaying='y',
            side='right',
            range=[0, max(10, df['power_cost'].max() * 1.2)]
        ),
        legend_title='Metric'
    )
    
    return fig

# NEW: Callback for Expert vs Neural Network usage
@app.callback(
    Output('expert-vs-neural', 'figure'),
    Input('decision-data', 'children')
)
def update_expert_vs_neural(data):
    if not data:
        return go.Figure()
    
    decisions = json.loads(data)
    if not decisions:
        return go.Figure()
    
    df = pd.DataFrame(decisions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create a column for decision source if not present
    if 'decision_source' not in df.columns:
        # Estimate based on expert_alignment
        df['decision_source'] = df.apply(lambda row: 
            'expert_table' if row.get('expert_alignment', False) else
            'q_table' if np.random.random() < 0.7 else
            'neural_network', axis=1)
    
    # Group by time and decision source
    source_counts = df.groupby([pd.Grouper(key='timestamp', freq='5min'), 'decision_source']).size().unstack(fill_value=0)
    
    # If columns are missing, add them with zeros
    for col in ['expert_table', 'q_table', 'neural_network']:
        if col not in source_counts.columns:
            source_counts[col] = 0
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for each decision source
    fig.add_trace(go.Scatter(
        x=source_counts.index,
        y=source_counts['expert_table'] if 'expert_table' in source_counts.columns else [0] * len(source_counts),
        mode='lines',
        name='Expert Table',
        stackgroup='one',
        line=dict(width=0, color='#006600')
    ))
    
    fig.add_trace(go.Scatter(
        x=source_counts.index,
        y=source_counts['q_table'] if 'q_table' in source_counts.columns else [0] * len(source_counts),
        mode='lines',
        name='Q-Table',
        stackgroup='one',
        line=dict(width=0, color='#0000cc')
    ))
    
    fig.add_trace(go.Scatter(
        x=source_counts.index,
        y=source_counts['neural_network'] if 'neural_network' in source_counts.columns else [0] * len(source_counts),
        mode='lines',
        name='Neural Network',
        stackgroup='one',
        line=dict(width=0, color='#cc6600')
    ))
    
    # Update layout
    fig.update_layout(
        title='Decision Sources Over Time',
        xaxis_title='Time',
        yaxis_title='Decision Count',
        legend_title='Source'
    )
    
    return fig

# NEW: Callback for discretized state visualization
@app.callback(
    Output('discretized-state', 'children'),
    Input('decision-data', 'children')
)
def update_discretized_state(data):
    if not data:
        return html.Div("No state data available")
    
    decisions = json.loads(data)
    if not decisions:
        return html.Div("No state data recorded")
    
    # Get latest decision
    latest = decisions[-1]
    state = latest.get('state', {})
    
    # Extract original state values (could be numeric or categorical)
    battery_original = state.get('battery', 'Unknown')
    temp_original = state.get('temperature', 'Unknown')
    threat_original = state.get('threat', 'Unknown')
    
    # Discretize battery
    if isinstance(battery_original, str):
        if any(level in battery_original for level in ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]):
            battery_zone = battery_original
        else:
            try:
                battery_value = float(battery_original.replace('%', ''))
                if battery_value <= 20:
                    battery_zone = "0-20%"
                elif battery_value <= 40:
                    battery_zone = "21-40%"
                elif battery_value <= 60:
                    battery_zone = "41-60%"
                elif battery_value <= 80:
                    battery_zone = "61-80%"
                else:
                    battery_zone = "81-100%"
            except:
                battery_zone = "unknown"
    else:
        try:
            battery_value = float(battery_original)
            if battery_value <= 20:
                battery_zone = "0-20%"
            elif battery_value <= 40:
                battery_zone = "21-40%"
            elif battery_value <= 60:
                battery_zone = "41-60%"
            elif battery_value <= 80:
                battery_zone = "61-80%"
            else:
                battery_zone = "81-100%"
        except:
            battery_zone = "unknown"
    
    # Discretize temperature
    if isinstance(temp_original, str):
        if temp_original in ["Safe", "Warning", "Critical"]:
            temp_zone = temp_original
        else:
            try:
                temp_value = float(temp_original.replace('°C', ''))
                if temp_value <= 55:
                    temp_zone = "Safe"
                elif temp_value <= 70:
                    temp_zone = "Warning"
                else:
                    temp_zone = "Critical"
            except:
                temp_zone = "unknown"
    else:
        try:
            temp_value = float(temp_original)
            if temp_value <= 55:
                temp_zone = "Safe"
            elif temp_value <= 70:
                temp_zone = "Warning"
            else:
                temp_zone = "Critical"
        except:
            temp_zone = "unknown"
    
    # Discretize threat
    if isinstance(threat_original, str):
        if threat_original in ["Normal", "Confirming", "Confirmed"]:
            threat_zone = threat_original
        else:
            threat_zone = "unknown"
    else:
        try:
            threat_value = int(threat_original)
            if threat_value == 0:
                threat_zone = "Normal"
            elif threat_value == 1:
                threat_zone = "Confirming"
            else:
                threat_zone = "Confirmed"
        except:
            threat_zone = "unknown"
    
    # Get lookup table key
    lookup_key = f"{battery_zone}, {temp_zone}, {threat_zone}"
    
    # Create discretized state visualization with lookup key info
    return html.Div([
        html.H4("Current Discretized State"),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
            # Temperature zone
            html.Div([
                html.Div("Temperature Zone", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div(temp_zone, style={
                    'backgroundColor': '#00aa00' if temp_zone == "Safe" else
                                     '#aaaa00' if temp_zone == "Warning" else
                                     '#aa0000',
                    'color': 'white',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }),
                html.Div(f"Raw: {temp_original}", style={'fontSize': '10px', 'textAlign': 'center', 'marginTop': '3px'})
            ], style={'width': '30%'}),
            
            # Battery zone
            html.Div([
                html.Div("Battery Zone", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div(battery_zone, style={
                    'backgroundColor': '#006600' if battery_zone in ['61-80%', '81-100%'] else
                                     '#999900' if battery_zone == '41-60%' else
                                     '#cc3300' if battery_zone in ['0-20%', '21-40%'] else '#888888',
                    'color': 'white',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }),
                html.Div(f"Raw: {battery_original}", style={'fontSize': '10px', 'textAlign': 'center', 'marginTop': '3px'})
            ], style={'width': '30%'}),
            
            # Threat state
            html.Div([
                html.Div("Threat State", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div(threat_zone, style={
                    'backgroundColor': '#006600' if threat_zone == "Normal" else
                                     '#cc6600' if threat_zone == "Confirming" else
                                     '#cc0000',
                    'color': 'white',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                }),
                html.Div(f"Raw: {threat_original}", style={'fontSize': '10px', 'textAlign': 'center', 'marginTop': '3px'})
            ], style={'width': '30%'})
        ]),
        
        # Lookup table key visualization
        html.Div([
            html.Div("Lookup Table Key:", style={'fontWeight': 'bold', 'marginTop': '15px'}),
            html.Div(lookup_key, style={
                'backgroundColor': '#f0f0f0',
                'padding': '10px',
                'borderRadius': '5px',
                'textAlign': 'center',
                'fontWeight': 'bold',
                'marginTop': '5px',
                'border': '1px solid #666'
            })
        ]),
        
        # Expert rule visualization
        html.Div([
            html.Div("Active Expert Rules:", style={'fontWeight': 'bold', 'marginTop': '15px', 'marginBottom': '5px'}),
            html.Ul([
                html.Li(f"Temperature: {get_temp_rule(temp_zone)}", 
                       style={'color': '#00aa00' if temp_zone == "Safe" else
                                      '#aaaa00' if temp_zone == "Warning" else
                                      '#aa0000'}),
                html.Li(f"Battery: {get_battery_rule(battery_zone)}",
                       style={'color': '#006600' if battery_zone in ['61-80%', '81-100%'] else
                                      '#999900' if battery_zone == '41-60%' else
                                      '#cc3300'}),
                html.Li(f"Algorithm: {get_algorithm_rule(latest.get('action_label', 'Unknown'))}",
                       style={'fontWeight': 'bold'})
            ])
        ])
    ], style={'backgroundColor': '#f9f9f9', 'padding': '15px', 'borderRadius': '5px'})

# Helper functions for expert rules
def get_temp_rule(temp_zone):
    rules = {
        'cold': 'All algorithms allowed',
        'cool': 'All algorithms allowed',
        'warm': 'TST with caution',
        'hot': 'Avoid TST algorithm',
        'danger': 'Emergency cooling required'
    }
    return rules.get(temp_zone, 'Unknown rule')

def get_power_rule(power):
    if power < 4.5:
        return 'Efficient - all algorithms allowed'
    elif power < 6.0:
        return 'Moderate - TST usage limited'
    else:
        return 'Excessive - power reduction required'

def get_algorithm_rule(algorithm):
    rules = {
        'No DDoS': 'Preserving critical systems',  # Updated label
        'XGBoost': 'Balanced detection efficiency',
        'TST': 'Maximum detection capability'
    }
    return rules.get(algorithm, 'Unknown algorithm')

# Additional helper function for battery rules
def get_battery_rule(battery_zone):
    rules = {
        '0-20%': 'CRITICAL: Only No_DDoS allowed',
        '21-40%': 'LOW: Avoid TST algorithm',
        '41-60%': 'MEDIUM: All algorithms allowed with caution',
        '61-80%': 'HIGH: All algorithms allowed',
        '81-100%': 'FULL: All algorithms allowed'
    }
    return rules.get(battery_zone, 'Unknown battery status')

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
