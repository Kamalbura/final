# Create analysis and monitoring script
analysis_script = '''#!/usr/bin/env python3
"""
UAV DDoS-RL Analysis and Monitoring Script
Analyze training results and monitor production performance
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class UAVDDoSAnalyzer:
    def __init__(self):
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting parameters"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def analyze_training_results(self, model_path):
        """Analyze training results from saved model"""
        print("="*60)
        print("TRAINING ANALYSIS")
        print("="*60)
        
        # Load model data
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        metrics = model_data['training_metrics']
        q_table = np.array(model_data['q_table'])
        expert_lookup = model_data['expert_lookup']
        
        # Basic statistics
        print(f"Model trained on: {model_data.get('timestamp', 'Unknown')}")
        print(f"Training episodes: {len(metrics['episodes'])}")
        print(f"Final reward: {metrics['rewards'][-1]:.1f}")
        print(f"Final expert alignment: {metrics['expert_alignment'][-1]:.3f}")
        print(f"Total safety violations: {sum(metrics['safety_violations'])}")
        print(f"Average power consumption: {np.mean(metrics['power_consumption']):.1f}W")
        
        # Q-table analysis
        print(f"\\nQ-TABLE ANALYSIS:")
        print(f"   Shape: {q_table.shape}")
        print(f"   Value range: [{q_table.min():.2f}, {q_table.max():.2f}]")
        print(f"   Mean Q-value: {q_table.mean():.2f}")
        print(f"   Std Q-value: {q_table.std():.2f}")
        
        # Expert policy analysis
        expert_actions = list(expert_lookup.values())
        action_dist = pd.Series(expert_actions).value_counts()
        print(f"\\nEXPERT POLICY DISTRIBUTION:")
        action_labels = ["No_DDoS", "XGBoost", "TST"]
        for action, count in action_dist.items():
            print(f"   {action_labels[action]}: {count} states ({count/45*100:.1f}%)")
        
        # Create visualizations
        self._plot_training_progress(metrics)
        self._plot_q_table_heatmap(q_table)
        
        return model_data
    
    def analyze_decision_log(self, log_path):
        """Analyze production decision log"""
        print("\\n" + "="*60)
        print("PRODUCTION DECISION ANALYSIS")
        print("="*60)
        
        with open(log_path, 'r') as f:
            decision_log = json.load(f)
        
        if not decision_log:
            print("No decisions in log file")
            return
        
        df = pd.DataFrame(decision_log)
        
        # Basic statistics
        total_decisions = len(df)
        safety_violations = df['safety_violation'].sum()
        expert_alignments = df['expert_alignment'].sum()
        total_power = df['power_cost'].sum()
        
        print(f"Total decisions: {total_decisions}")
        print(f"Safety violations: {safety_violations} ({safety_violations/total_decisions*100:.1f}%)")
        print(f"Expert alignment: {expert_alignments} ({expert_alignments/total_decisions*100:.1f}%)")
        print(f"Total power consumed: {total_power:.1f}W")
        print(f"Average power per decision: {total_power/total_decisions:.2f}W")
        
        # Action distribution
        action_dist = df['action_label'].value_counts()
        print(f"\\nACTION DISTRIBUTION:")
        for action, count in action_dist.items():
            print(f"   {action}: {count} ({count/total_decisions*100:.1f}%)")
        
        # State analysis
        print(f"\\nSTATE DISTRIBUTION:")
        for col in ['battery', 'temperature', 'threat']:
            state_counts = df['state'].apply(lambda x: x[col]).value_counts()
            print(f"   {col.title()}:")
            for state, count in state_counts.items():
                print(f"     {state}: {count} ({count/total_decisions*100:.1f}%)")
        
        # Time analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        hourly_decisions = df.groupby('hour').size()
        
        print(f"\\nHOURLY DECISION PATTERN:")
        for hour, count in hourly_decisions.items():
            print(f"   Hour {hour:2d}: {count:3d} decisions")
        
        # Performance over time
        self._plot_production_performance(df)
        
        return df
    
    def compare_policies(self, model_path):
        """Compare trained policy with expert policy"""
        print("\\n" + "="*60)
        print("POLICY COMPARISON ANALYSIS")
        print("="*60)
        
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        q_table = np.array(model_data['q_table'])
        expert_lookup = model_data['expert_lookup']
        
        # State space definitions
        battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        temperatures = ["Safe", "Warning", "Critical"]
        threat_states = ["Normal", "Confirming", "Confirmed"]
        action_labels = ["No_DDoS", "XGBoost", "TST"]
        
        def get_state_index(battery, temp, threat):
            battery_idx = battery_levels.index(battery)
            temp_idx = temperatures.index(temp)
            threat_idx = threat_states.index(threat)
            return battery_idx * 9 + temp_idx * 3 + threat_idx
        
        # Compare policies
        comparison_results = []
        perfect_matches = 0
        
        for state_key, expert_action in expert_lookup.items():
            if isinstance(state_key, str):
                # Handle string keys
                battery, temp, threat = state_key.split('|')
            else:
                battery, temp, threat = state_key
            
            state_idx = get_state_index(battery, temp, threat)
            trained_action = np.argmax(q_table[state_idx])
            
            match = (trained_action == expert_action)
            if match:
                perfect_matches += 1
            
            comparison_results.append({
                'battery': battery,
                'temperature': temp,
                'threat': threat,
                'expert_action': expert_action,
                'trained_action': trained_action,
                'expert_label': action_labels[expert_action],
                'trained_label': action_labels[trained_action],
                'match': match,
                'q_values': q_table[state_idx].tolist()
            })
        
        # Results
        total_states = len(comparison_results)
        alignment_rate = perfect_matches / total_states
        
        print(f"Policy alignment: {perfect_matches}/{total_states} ({alignment_rate*100:.1f}%)")
        
        # Show mismatches
        mismatches = [r for r in comparison_results if not r['match']]
        if mismatches:
            print(f"\\n⚠️  POLICY MISMATCHES ({len(mismatches)} states):")
            print("   Battery    | Temp     | Threat     | Expert    | Trained   ")
            print("   " + "-"*65)
            for mismatch in mismatches:
                print(f"   {mismatch['battery']:9s} | {mismatch['temperature']:8s} | "
                      f"{mismatch['threat']:10s} | {mismatch['expert_label']:9s} | "
                      f"{mismatch['trained_label']:9s}")
        else:
            print("\\n✅ Perfect policy alignment - no mismatches!")
        
        return comparison_results
    
    def _plot_training_progress(self, metrics):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UAV DDoS-RL Training Progress', fontsize=16)
        
        # Rewards
        axes[0,0].plot(metrics['episodes'], metrics['rewards'])
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].grid(True)
        
        # Expert alignment
        axes[0,1].plot(metrics['episodes'], metrics['expert_alignment'])
        axes[0,1].set_title('Expert Alignment')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Alignment Rate')
        axes[0,1].set_ylim(0, 1.1)
        axes[0,1].grid(True)
        
        # Power consumption
        axes[1,0].plot(metrics['episodes'], metrics['power_consumption'])
        axes[1,0].set_title('Power Consumption per Episode')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Power (W)')
        axes[1,0].grid(True)
        
        # Epsilon decay
        axes[1,1].plot(metrics['episodes'], metrics['epsilon_values'])
        axes[1,1].set_title('Exploration Rate (Epsilon)')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Epsilon')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\\n📊 Training progress plots saved as 'training_progress.png'")
    
    def _plot_q_table_heatmap(self, q_table):
        """Plot Q-table as heatmap"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(q_table, cmap='RdYlGn', center=0, 
                   xticklabels=['No_DDoS', 'XGBoost', 'TST'],
                   yticklabels=[f'State {i}' for i in range(len(q_table))])
        plt.title('Q-Table Values Heatmap')
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.tight_layout()
        plt.savefig('q_table_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Q-table heatmap saved as 'q_table_heatmap.png'")
    
    def _plot_production_performance(self, df):
        """Plot production performance over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Production Performance Analysis', fontsize=16)
        
        # Actions over time
        action_timeline = df.set_index('timestamp')['action_label']
        action_counts = action_timeline.resample('H').value_counts().unstack(fill_value=0)
        action_counts.plot(kind='area', stacked=True, ax=axes[0,0])
        axes[0,0].set_title('Actions Over Time')
        axes[0,0].set_ylabel('Count per Hour')
        
        # Power consumption over time
        power_timeline = df.set_index('timestamp')['power_cost']
        power_hourly = power_timeline.resample('H').sum()
        power_hourly.plot(ax=axes[0,1])
        axes[0,1].set_title('Power Consumption Over Time')
        axes[0,1].set_ylabel('Power (W) per Hour')
        
        # Safety violations
        safety_timeline = df.set_index('timestamp')['safety_violation']
        safety_hourly = safety_timeline.resample('H').sum()
        safety_hourly.plot(ax=axes[1,0], color='red')
        axes[1,0].set_title('Safety Violations Over Time')
        axes[1,0].set_ylabel('Violations per Hour')
        
        # Expert alignment
        expert_timeline = df.set_index('timestamp')['expert_alignment']
        expert_hourly = expert_timeline.resample('H').mean()
        expert_hourly.plot(ax=axes[1,1], color='green')
        axes[1,1].set_title('Expert Alignment Over Time')
        axes[1,1].set_ylabel('Alignment Rate')
        axes[1,1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig('production_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Production performance plots saved as 'production_performance.png'")
    
    def generate_report(self, model_path, decision_log_path=None):
        """Generate comprehensive analysis report"""
        print("="*80)
        print("UAV POWER-AWARE DDoS-RL COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Training analysis
        model_data = self.analyze_training_results(model_path)
        
        # Policy comparison
        policy_comparison = self.compare_policies(model_path)
        
        # Production analysis (if log available)
        if decision_log_path:
            try:
                production_df = self.analyze_decision_log(decision_log_path)
            except FileNotFoundError:
                print(f"\\n⚠️  Decision log not found: {decision_log_path}")
                production_df = None
        else:
            production_df = None
        
        # Summary
        print("\\n" + "="*80)
        print("FINAL ASSESSMENT")
        print("="*80)
        
        training_metrics = model_data['training_metrics']
        final_alignment = training_metrics['expert_alignment'][-1]
        total_violations = sum(training_metrics['safety_violations'])
        
        print(f"✅ Training Status: {'SUCCESS' if final_alignment > 0.9 else 'NEEDS IMPROVEMENT'}")
        print(f"✅ Safety Status: {'EXCELLENT' if total_violations == 0 else 'VIOLATIONS DETECTED'}")
        print(f"✅ Expert Alignment: {final_alignment:.1%}")
        print(f"✅ Policy Consistency: {'MAINTAINED' if len([p for p in policy_comparison if not p['match']]) == 0 else 'SOME DEVIATIONS'}")
        
        if production_df is not None:
            prod_safety = production_df['safety_violation'].sum()
            prod_alignment = production_df['expert_alignment'].mean()
            print(f"✅ Production Safety: {'EXCELLENT' if prod_safety == 0 else f'{prod_safety} VIOLATIONS'}")
            print(f"✅ Production Alignment: {prod_alignment:.1%}")
        
        print("\\n🎯 SYSTEM READY FOR DEPLOYMENT" if final_alignment > 0.9 and total_violations == 0 else "\\n⚠️  SYSTEM NEEDS REVIEW")

# MAIN ANALYSIS SCRIPT
if __name__ == "__main__":
    analyzer = UAVDDoSAnalyzer()
    
    try:
        # Generate comprehensive report
        analyzer.generate_report(
            model_path='trained_uav_ddos_model.json',
            decision_log_path='test_decisions.json'  # Optional
        )
        
    except FileNotFoundError as e:
        print(f"❌ Required file not found: {e}")
        print("Please ensure you have run the training script first.")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
'''

# Save analysis script
with open('uav_ddos_analysis.py', 'w') as f:
    f.write(analysis_script)

print("✅ Complete analysis script saved to: uav_ddos_analysis.py")