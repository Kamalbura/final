#!/usr/bin/env python3
"""
Visualizes the expert lookup table in various formats
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def create_expert_lookup_table():
    """Create lookup table from user-provided expert knowledge"""
    entries = []
    
    # Battery levels
    battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
    temperatures = ["Safe", "Warning", "Critical"]
    threat_states = ["Normal", "Confirming", "Confirmed"]
    action_labels = ["No_DDoS", "XGBoost", "TST"]
    
    # Add entries from user-provided table
    for battery in battery_levels:
        for temp in temperatures:
            for threat in threat_states:
                # Default action is No DDoS
                action = 0
                
                # Critical conditions always use No DDoS
                if battery == "0-20%" or temp == "Critical":
                    action = 0
                # 21-40% battery
                elif battery == "21-40%":
                    if threat in ["Confirming", "Confirmed"] and temp != "Critical":
                        action = 1  # XGBoost
                    else:
                        action = 0  # No DDoS
                # 41-60% battery or higher
                elif battery in ["41-60%", "61-80%", "81-100%"]:
                    if threat == "Confirming" and temp != "Critical":
                        action = 2  # TST
                    elif threat != "Normal" and temp != "Critical":
                        action = 1  # XGBoost
                    elif temp != "Critical":
                        action = 1  # XGBoost
                    else:
                        action = 0  # No DDoS
                
                entries.append({
                    'Battery_Level': battery,
                    'Temperature': temp,
                    'Threat_State': threat,
                    'Action': action,
                    'Action_Label': action_labels[action]
                })
    
    return pd.DataFrame(entries)

def create_heatmap_visualization(df):
    """Create heatmap visualizations for the lookup table"""
    os.makedirs('visualizations', exist_ok=True)
    
    # Create color map
    colors = ['green', 'blue', 'red']
    cmap = LinearSegmentedColormap.from_list('action_colors', colors, N=3)
    
    # For each threat state
    for threat in df['Threat_State'].unique():
        threat_data = df[df['Threat_State'] == threat]
        
        # Pivot table for heatmap
        heatmap_data = threat_data.pivot_table(
            index='Battery_Level', 
            columns='Temperature',
            values='Action'
        )
        
        # Ensure consistent order
        heatmap_data = heatmap_data.reindex(
            index=["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"],
            columns=["Safe", "Warning", "Critical"]
        )
        
        # Convert to integer explicitly to avoid format error
        heatmap_data = heatmap_data.astype(int)
        
        # Create heatmap plot
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            heatmap_data, 
            cmap=cmap, 
            annot=True, 
            fmt="d",  # This now works because we converted to int
            cbar_kws={'ticks': [0, 1, 2], 'label': 'Action'}
        )
        
        plt.title(f'Actions for Threat State: {threat}', fontsize=16)
        
        # Add second annotation layer with action labels
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                action = int(heatmap_data.iloc[i, j])
                ax.text(j + 0.5, i + 0.5, ["No_DDoS", "XGBoost", "TST"][action], 
                        ha="center", va="center", color="white", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/lookup_heatmap_threat_{threat}.png')
        plt.close()
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define coordinates for each state
    x = []  # Battery
    y = []  # Temperature
    z = []  # Threat
    colors = []  # Action
    
    battery_map = {"0-20%": 0, "21-40%": 1, "41-60%": 2, "61-80%": 3, "81-100%": 4}
    temp_map = {"Safe": 0, "Warning": 1, "Critical": 2}
    threat_map = {"Normal": 0, "Confirming": 1, "Confirmed": 2}
    action_colors = ['green', 'blue', 'red']
    
    for _, row in df.iterrows():
        x.append(battery_map[row['Battery_Level']])
        y.append(temp_map[row['Temperature']])
        z.append(threat_map[row['Threat_State']])
        colors.append(action_colors[int(row['Action'])])  # Ensure Action is int
    
    # Create scatter plot
    ax.scatter(x, y, z, c=colors, s=100)
    
    # Set labels and ticks
    ax.set_xlabel('Battery Level')
    ax.set_ylabel('Temperature')
    ax.set_zlabel('Threat State')
    
    ax.set_xticks(list(battery_map.values()))
    ax.set_xticklabels(list(battery_map.keys()))
    ax.set_yticks(list(temp_map.values()))
    ax.set_yticklabels(list(temp_map.keys()))
    ax.set_zticks(list(threat_map.values()))
    ax.set_zticklabels(list(threat_map.keys()))
    
    plt.title('3D Visualization of Lookup Table', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/lookup_table_3d.png')
    plt.close()

def generate_lookup_table_summary(df):
    """Generate summary statistics for the lookup table"""
    total_entries = len(df)
    action_counts = df['Action'].value_counts().sort_index()
    
    print("="*60)
    print("LOOKUP TABLE SUMMARY")
    print("="*60)
    print(f"Total entries: {total_entries}")
    
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"Action {action} ({df['Action_Label'][df['Action'] == action].iloc[0]}): "
              f"{count} entries ({count/total_entries*100:.1f}%)")
    
    print("\nBattery Level Breakdown:")
    for battery in ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]:
        subset = df[df['Battery_Level'] == battery]
        print(f"{battery}: {len(subset)} entries")
        for action in [0, 1, 2]:
            count = len(subset[subset['Action'] == action])
            if count > 0:
                print(f"  - Action {action}: {count} entries ({count/len(subset)*100:.1f}%)")
    
    print("\nCritical Conditions:")
    critical = df[(df['Battery_Level'] == "0-20%") | (df['Temperature'] == "Critical")]
    print(f"Total critical entries: {len(critical)}")
    print(f"No DDoS actions in critical: {len(critical[critical['Action'] == 0])}")
    if len(critical[critical['Action'] != 0]) > 0:
        print("WARNING: Some critical conditions don't use No DDoS!")
    
    # Confirming threat analysis
    confirming = df[df['Threat_State'] == "Confirming"]
    confirming_tst = confirming[confirming['Action'] == 2]
    print("\nConfirming Threat Analysis:")
    print(f"Total confirming threat entries: {len(confirming)}")
    print(f"TST used in confirming threats: {len(confirming_tst)} ({len(confirming_tst)/len(confirming)*100:.1f}%)")
    
    # Generate verification points
    print("\nVerification Points:")
    check_states = [
        ("0-20%", "Safe", "Confirming"),
        ("21-40%", "Safe", "Confirming"),
        ("41-60%", "Safe", "Confirming"),
        ("81-100%", "Warning", "Confirmed"),
        ("61-80%", "Critical", "Normal")
    ]
    
    for battery, temp, threat in check_states:
        entry = df[(df['Battery_Level'] == battery) & 
                  (df['Temperature'] == temp) & 
                  (df['Threat_State'] == threat)]
        if len(entry) > 0:
            action = entry['Action'].iloc[0]
            label = entry['Action_Label'].iloc[0]
            print(f"({battery}, {temp}, {threat}) → Action {action} ({label})")

if __name__ == "__main__":
    print("Generating expert lookup table visualizations...")
    
    # Create lookup table
    lookup_df = create_expert_lookup_table()
    
    # Ensure the Action column is integer type
    lookup_df['Action'] = lookup_df['Action'].astype(int)
    
    # Create visualizations
    create_heatmap_visualization(lookup_df)
    
    # Generate summary
    generate_lookup_table_summary(lookup_df)
    
    # Save as CSV for reference
    lookup_df.to_csv('visualizations/lookup_table.csv', index=False)
    
    print("\nVisualizations created in 'visualizations' directory:")
    print("- lookup_heatmap_threat_*.png: Heatmaps for each threat state")
    print("- lookup_table_3d.png: 3D visualization of entire lookup table")
    print("- lookup_table.csv: CSV export of the lookup table")
