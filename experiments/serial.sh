#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --mem=500M

# Run your code here
wandb agent --count 1 xinyu-zhang/accuracy-concurvity-curve-NAMs/toqj55ak
