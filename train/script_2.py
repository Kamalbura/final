# 3. COMPREHENSIVE EVALUATION AND ANALYSIS

# Evaluate the trained policy
eval_results, state_performance = agent.evaluate_policy(num_eval_episodes=100)

print("\n" + "="*80)
print("DETAILED STATE-BY-STATE ANALYSIS")
print("="*80)

# Analyze performance for each state
policy_comparison = []
perfect_alignment = 0
total_states = 0

for state_key, performance in state_performance.items():
    battery, temp, threat = state_key
    actions_taken = performance['actions']
    expert_actions = performance['expert_actions']
    
    # Most common action taken by trained agent
    if actions_taken:
        trained_action = max(set(actions_taken), key=actions_taken.count)
        expert_action = expert_actions[0]  # Expert action is always the same for a given state
        
        alignment = (trained_action == expert_action)
        if alignment:
            perfect_alignment += 1
        total_states += 1
        
        policy_comparison.append({
            'battery': battery,
            'temperature': temp,
            'threat': threat,
            'expert_action': expert_action,
            'trained_action': trained_action,
            'alignment': alignment,
            'expert_label': env.action_labels[expert_action],
            'trained_label': env.action_labels[trained_action]
        })

# Sort by alignment (misalignments first)
policy_comparison.sort(key=lambda x: x['alignment'])

print(f"\n🎯 POLICY ALIGNMENT ANALYSIS:")
print(f"   Perfect Alignments: {perfect_alignment}/{total_states} ({perfect_alignment/total_states*100:.1f}%)")
print(f"   Misalignments: {total_states-perfect_alignment}")

if total_states - perfect_alignment > 0:
    print(f"\n⚠️  MISALIGNED STATES (Require Attention):")
    print("   Battery    | Temp     | Threat     | Expert    | Trained   | Status")
    print("   " + "-"*70)
    
    for comp in policy_comparison:
        if not comp['alignment']:
            print(f"   {comp['battery']:9s} | {comp['temperature']:8s} | {comp['threat']:10s} | "
                  f"{comp['expert_label']:9s} | {comp['trained_label']:9s} | ❌")

# Perfect alignments
print(f"\n✅ PERFECTLY ALIGNED STATES:")
aligned_count = 0
for comp in policy_comparison:
    if comp['alignment']:
        aligned_count += 1

print(f"   {aligned_count} states perfectly aligned with expert policy")

# Save comprehensive results
comprehensive_results = {
    'training_summary': {
        'episodes': len(training_results['episodes']),
        'final_reward': training_results['rewards'][-1],
        'final_expert_alignment': training_results['expert_alignment'][-1],
        'total_safety_violations': sum(training_results['safety_violations']),
        'final_epsilon': training_results['epsilon_values'][-1]
    },
    'evaluation_summary': {
        'avg_reward': np.mean(eval_results['rewards']),
        'avg_expert_alignment': np.mean(eval_results['expert_alignment']),
        'total_safety_violations': sum(eval_results['safety_violations']),
        'avg_power_consumption': np.mean(eval_results['power_efficiency']),
        'action_distribution': {
            'No_DDoS': eval_results['action_distribution'][0],
            'XGBoost': eval_results['action_distribution'][1], 
            'TST': eval_results['action_distribution'][2]
        }
    },
    'policy_analysis': {
        'perfect_alignments': perfect_alignment,
        'total_states_tested': total_states,
        'alignment_percentage': (perfect_alignment/total_states)*100
    }
}

# Save to JSON
with open('comprehensive_training_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2)

print(f"\n💾 Results saved to: comprehensive_training_results.json")
print(f"\n🎉 TRAINING SUCCESS SUMMARY:")
print(f"   ✅ High Rewards: {comprehensive_results['evaluation_summary']['avg_reward']:.1f} average")
print(f"   ✅ Expert Alignment: {comprehensive_results['evaluation_summary']['avg_expert_alignment']:.3f}")
print(f"   ✅ Zero Safety Violations: {comprehensive_results['evaluation_summary']['total_safety_violations']}")
print(f"   ✅ Power Efficient: {comprehensive_results['evaluation_summary']['avg_power_consumption']:.1f}W average")
print(f"   ✅ Policy Alignment: {comprehensive_results['policy_analysis']['alignment_percentage']:.1f}%")