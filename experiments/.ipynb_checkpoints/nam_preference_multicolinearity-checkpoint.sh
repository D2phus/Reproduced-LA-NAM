#!/bin/bash
# SBATCH --time=00:20:00
# SBATCH --mem=1G
#SBATCH --job-name=multicolinearity-preference-hardcoded
# SBATCH --output=mutlicolinearity-preference-array-hardcoded_%a.out
# SBATCH --array=0-4

case $SLURM_ARRAY_TASK_ID in
   0)  SCALE=0.1 ;;
   1)  SCALE=0.5 ;;
   2)  SCALE=1 ;;
   3)  SCALE=2 ;;
   4)  SCALE=10 ;;
esac

srun python nam_preference_multicolinearity.py --scale=$SCALE 
