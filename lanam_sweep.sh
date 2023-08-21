#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=5G
#SBATCH --job-name=lanam-sweep-array
#SBATCH --output=lanam-sweep-array_%a.out
#SBATCH --array=0-4

case $SLURM_ARRAY_TASK_ID in
   0)  ACT_CLS='gelu' ;;
   1)  ACT_CLS='leakyrelu' ;;
   2)  ACT_CLS='exu' ;;
   3)  ACT_CLS='leakyrelu' ;;
   4)  ACT_CLS='elu' ;;
esac

srun --mem=700M --time=30:00:00 python3 lanam_sweep.py --activation_cls=$ACT_CLS 