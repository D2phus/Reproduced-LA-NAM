#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=700M
#SBATCH --job-name=lanam-sweep-array-hardcoded
#SBATCH --output=lanam-sweep-array-hardcoded_%a.out
#SBATCH --array=0-4

case $SLURM_ARRAY_TASK_ID in
   0)  ACT_CLS='relu' ;;
   1)  ACT_CLS='gelu' ;;
   2)  ACT_CLS='exu' ;;
   3)  ACT_CLS='leakyrelu' ;;
   4)  ACT_CLS='elu' ;;
esac

srun python lanam_sweep.py --activation_cls=$ACT_CLS