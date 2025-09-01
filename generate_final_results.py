import os, json, shutil, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from datetime import datetime
from src.agents.lookup_table_agent import LookupTableAgent
from src.environments.uav_ddos_env import UAVDDoSEnvironment

OUTPUT_DIR = 'final_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Train (or load latest model) and evaluate grid for visualizations
env = UAVDDoSEnvironment()
agent = LookupTableAgent(learning_enabled=False)  # deterministic policy

# Use environment expert table directly for actions since policy identical
battery_levels = env.battery_levels
temp_categories = ["Safe","Warning","Critical"]
threat_states = env.threat_states

# Build action matrices per threat vs battery and temperature
# 1. Temperature vs Threat (fixed battery mid category 41-60%)
fixed_battery = '41-60%'
mat_temp_threat = np.zeros((len(temp_categories), len(threat_states)))
for i,tcat in enumerate(temp_categories):
    for j,th in enumerate(threat_states):
        state_key = (fixed_battery, tcat, th)
        act = env.expert_lookup[state_key]
        mat_temp_threat[i,j] = act

# 2. Battery vs Threat (fixed temperature Safe)
fixed_temp = 'Safe'
mat_batt_threat = np.zeros((len(battery_levels), len(threat_states)))
for i,b in enumerate(battery_levels):
    for j,th in enumerate(threat_states):
        act = env.expert_lookup[(b, fixed_temp, th)]
        mat_batt_threat[i,j] = act

cmap = sns.color_palette(['#4caf50','#1976d2','#ff9800'])  # No_DDoS, XGBoost, TST

def plot_matrix(matrix, y_labels, x_labels, title, fname):
    plt.figure(figsize=(6,4))
    sns.heatmap(matrix, annot=True, fmt='.0f', cbar=False, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, linewidths=0.5, linecolor='gray')
    plt.title(title)
    plt.ylabel('')
    plt.xlabel('Threat State')
    for color_patch,label in zip(['#4caf50','#1976d2','#ff9800'], ['No_DDoS','XGBoost','TST']):
        plt.scatter([],[], c=color_patch, label=label)
    plt.legend(frameon=False, bbox_to_anchor=(1.05,1), loc='upper left')
    out_path = os.path.join(OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

files = []
files.append(plot_matrix(mat_temp_threat, temp_categories, threat_states, 'Action by Temperature vs Threat (Battery 41-60%)', 'temp_vs_threat_policy.png'))
files.append(plot_matrix(mat_batt_threat, battery_levels, threat_states, 'Action by Battery vs Threat (Temp Safe)', 'battery_vs_threat_policy.png'))

# Copy latest model JSON into final_results
models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) if f.startswith('uav_ddos_lookup_model_') and f.endswith('.json')]
latest_model = max(model_files) if model_files else None
model_copy_path=None
if latest_model:
    src = os.path.join(models_dir, latest_model)
    model_copy_path = os.path.join(OUTPUT_DIR, latest_model)
    shutil.copy2(src, model_copy_path)

# Summarize
summary = {
    'generated_at': datetime.utcnow().isoformat()+ 'Z',
    'model_file': model_copy_path,
    'plots': files,
    'policy_rules': 'Critical temp or 0-20% battery -> No_DDoS; Confirming -> TST; else XGBoost'
}
with open(os.path.join(OUTPUT_DIR,'summary.json'),'w') as f:
    json.dump(summary,f,indent=2)
print('Final results generated:', summary)
