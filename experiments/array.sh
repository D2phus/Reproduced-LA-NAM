#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=500M
#SBATCH --job-name=pi-array-hardcoded
#SBATCH --output=pi-array-hardcoded_%a.out
#SBATCH --array=0-5

case $SLURM_ARRAY_TASK_ID in
   0)  EXP=5 ;;
   1)  EXP=4  ;;
   2)  EXP=3  ;;
   3)  EXP=2  ;;
   4)  EXP=1  ;;
   5)  EXP=0  ;;
esac

srun python lanam_interaction.py --exp=$EXP 