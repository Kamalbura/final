#!/usr/bin/env python3
"""
Script to organize the codebase structure for the UAV DDoS defense system
"""

import os
import shutil
import sys

def create_directory_structure():
    """Create the directory structure for the organized project"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories to create
    directories = [
        'src',
        'src/agents',
        'src/environments',
        'src/utils',
        'models',
        'visualizations',
        'notebooks',
        'examples',
        'tests'
    ]
    
    # Create directories
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created directory: {full_path}")
        else:
            print(f"Directory already exists: {full_path}")
    
    return base_dir

def create_init_files(base_dir):
    """Create __init__.py files to make directories importable packages"""
    package_dirs = [
        'src',
        'src/agents',
        'src/environments',
        'src/utils',
    ]
    
    for directory in package_dirs:
        init_file = os.path.join(base_dir, directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# This file makes the directory a Python package\n')
            print(f"Created __init__.py in {directory}")

def create_readme(base_dir):
    """Create a README.md file with project information"""
    readme_path = os.path.join(base_dir, 'README.md')
    if not os.path.exists(readme_path):
        readme_content = """# UAV DDoS Defense System

## Overview
This project implements a lookup table-based reinforcement learning approach for UAV DDoS defense that optimizes power consumption while maintaining security.

## Features
- Expert knowledge-based lookup table initialization
- Q-learning for policy refinement
- Safety-first decision making
- Power and thermal-aware operation
- Efficient resource management

## Directory Structure
- `src/`: Core implementation code
  - `agents/`: Agent implementations
  - `environments/`: UAV simulation environment
  - `utils/`: Utility functions
- `models/`: Trained models and lookup tables
- `visualizations/`: Generated charts and visualizations
- `notebooks/`: Jupyter notebooks for experimentation
- `examples/`: Example scripts and demos
- `tests/`: Test scripts

## Usage
```python
# Quick start
from src.agents.lookup_table_agent import LookupTableAgent
from src.environments.uav_ddos_env import UAVDDoSEnvironment

# Initialize environment and agent
env = UAVDDoSEnvironment()
agent = LookupTableAgent()

# Train the agent
training_metrics = agent.train(env, episodes=200)

# Use the trained agent for decision making
state = {'battery': 75, 'temperature': 52, 'threat': 1}
action = agent.make_decision(state)
```

## License
MIT License
"""
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"Created README.md")

def main():
    """Main function to organize the codebase"""
    print("Organizing UAV DDoS Defense codebase...")
    
    base_dir = create_directory_structure()
    create_init_files(base_dir)
    create_readme(base_dir)
    
    print("\nCodebase organization complete!")
    print("Next steps:")
    print("1. Move existing files to their appropriate directories")
    print("2. Update imports in all files to reflect new structure")
    print("3. Run tests to ensure everything still works")

if __name__ == "__main__":
    main()
