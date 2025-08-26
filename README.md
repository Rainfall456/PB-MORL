# PB-MORL
Code for "Population-Based Multi-Objective Reinforcement Learning with Information Sharing and Differentiation".

# Dependency
python 3.8.13, pytorch 1.8.2, gym 0.23.1, mujoco-py 2.1.2.14, pymoo 0.5.0, numpy 1.21.5, matplotlib 3.7.5

# Train
python PB-MORL.py --env 'MO_hopper-v0' --seed 0

# Test model and Visualize its PF approximation
python test_model.py --env 'MO_half_cheetah-v0'  

Since the final model is sometimes not optimal, models used for testing were selected from the later stages of training.
