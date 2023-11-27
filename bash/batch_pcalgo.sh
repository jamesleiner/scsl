#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=1-35
VAR=$((SLURM_ARRAY_TASK_ID))
echo $VAR

VARIABLE=$(head -"$VAR" bash/params_pcalgo.txt | tail -1)
echo $VARIABLE
stringarray=($VARIABLE)

module load python
module load anaconda3
conda activate hrt-funs
python run_pcalgo_cont.py --alpha ${stringarray[2]} --label  ${stringarray[0]}  --model_type ${stringarray[1]} 