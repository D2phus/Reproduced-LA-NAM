#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=700M
#SBATCH --job-name=nam-sweep-array-hardcoded
#SBATCH --output=nam-sweep-array-hardcoded_%a.out
#SBATCH --array=0

case $SLURM_ARRAY_TASK_ID in
   0)  ACT_CLS='elu'
       HID_SIZ=64
   ;;
esac

srun python nam_sweep.py --activation_cls=$ACT_CLS --hidden_sizes=$HID_SIZ
