# Evaluate the trained Q-table vs initial expert policy
print("=== EVALUATION: TRAINED Q-TABLE VS EXPERT POLICY ===")

# Test the final policy on a set of representative states
test_states = [
    ("0-20%", "Safe", "Normal"),      # Low battery, should be No DDoS
    ("0-20%", "Critical", "Confirming"),  # Critical conditions
    ("81-100%", "Safe", "Confirming"),    # Good conditions for TST
    ("41-60%", "Warning", "Confirmed"),   # Medium battery, confirmed threat
    ("61-80%", "Safe", "Normal"),         # Normal traffic with good battery
]

print("\nPolicy comparison for key states:")
print("State (Battery, Temp, Threat) | Expert Action | Trained Action | Q-values")
print("-" * 80)

for battery, temp, threat in test_states:
    state_idx = get_state_index(battery, temp, threat)
    
    # Get expert action from original lookup
    expert_action = None
    for entry in expert_lookup:
        if entry['battery'] == battery and entry['temperature'] == temp and entry['threat'] == threat:
            expert_action = entry['action']
            break
    
    # Get trained action
    trained_action = np.argmax(Q_table[state_idx])
    q_values = Q_table[state_idx]
    
    action_names = ["No DDoS", "XGBoost", "TST"]
    print(f"({battery:6s}, {temp:8s}, {threat:10s}) | {action_names[expert_action]:8s}  | {action_names[trained_action]:11s} | {q_values}")

# Analyze Q-table changes
print(f"\n=== Q-TABLE EVOLUTION ANALYSIS ===")

# Count how many Q-values changed significantly from initial values
initial_expert_values = (Q_table == 10.0).sum()
initial_default_values = (Q_table == -1.0).sum()
changed_values = ((Q_table != 10.0) & (Q_table != -1.0)).sum()

print(f"Q-values still at expert initialization (10.0): {initial_expert_values}")
print(f"Q-values still at default (-1.0): {initial_default_values}")
print(f"Q-values modified by learning: {changed_values}")

# Show distribution of final Q-values
print(f"\nQ-value statistics:")
print(f"Min Q-value: {Q_table.min():.2f}")
print(f"Max Q-value: {Q_table.max():.2f}")
print(f"Mean Q-value: {Q_table.mean():.2f}")
print(f"Std Q-value: {Q_table.std():.2f}")

# Save results to CSV for analysis
results_df = pd.DataFrame({
    'episode': range(num_episodes),
    'reward': episode_rewards,
    'power_consumption': episode_powers,
    'detections': episode_detections
})

results_df.to_csv('ql_training_results.csv', index=False)

print(f"\n=== TRAINING PERFORMANCE SUMMARY ===")
print(f"Episodes: {num_episodes}")
print(f"Final exploration rate: {epsilon:.3f}")
print(f"Total Q-values updated: {changed_values} out of {Q_table.size}")
print(f"Learning efficiency: Expert knowledge provided strong initialization")
print(f"Power efficiency: Average {np.mean(episode_powers):.1f}W per episode")
print(f"Detection rate: {np.mean(episode_detections)/20*100:.1f}% per timestep")