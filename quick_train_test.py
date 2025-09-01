from src.environments.uav_ddos_env import UAVDDoSEnvironment
from src.agents.lookup_table_agent import LookupTableAgent

env = UAVDDoSEnvironment()
agent = LookupTableAgent()
metrics = agent.train(env, num_episodes=5, max_steps=150)
print('Episodes:', metrics['episodes'][-1]+1 if metrics.get('episodes') else 0)
print('Last reward:', metrics['rewards'][-1] if metrics.get('rewards') else None)
print('Last expert alignment:', metrics['expert_alignment'][-1] if metrics.get('expert_alignment') else None)
print('Safety violations (env):', env.safety_violations)
