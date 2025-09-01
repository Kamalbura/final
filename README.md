# UAV DDoS Defense System

## Overview
This project implements a lookup table-based reinforcement learning approach for UAV DDoS defense that optimizes power consumption, security, and thermal performance.

## Key Features
- **Expert-Initialized Lookup Table**: Starts with expert knowledge for immediate safety
- **Q-Learning Refinement**: Learns to improve decisions through experience
- **Thermal-Aware**: Prevents overheating during DDoS detection
- **Power-Optimized**: Balances security needs with battery constraints
- **Safety Guarantees**: Hard constraints prevent unsafe operations

## Project Structure

## Training Commands
To train the model, use the following command:

```bash
python train.py --config config.yaml
```
This command will start the training process using the parameters specified in `config.yaml`. Make sure to adjust the parameters in the configuration file according to your needs before starting the training.

For a comprehensive training setup, consider the following commands:

```bash
# To train with a specific number of episodes
python train.py --config config.yaml --episodes 1000

# To resume training from a checkpoint
python train.py --config config.yaml --resume --checkpoint path_to_checkpoint

# To evaluate the model
python evaluate.py --config config.yaml --model path_to_trained_model
```

Ensure that you have the necessary dependencies installed and your environment is correctly set up before executing these commands.
