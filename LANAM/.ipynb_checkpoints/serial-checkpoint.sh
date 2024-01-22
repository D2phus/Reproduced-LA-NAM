#!/bin/bash -l
# SBATCH --time=01:00:00
# SBATCH --mem=1200M

# Run your code here
wandb agent --count 15 xinyu-zhang/kernel-based-concurvity-regularization-in-NAMs/sfrr9cq1

