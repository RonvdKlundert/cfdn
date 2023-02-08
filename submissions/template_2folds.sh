#!/bin/bash
#SBATCH -t ---time---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=---n---

#SBATCH --array=0-1
#SBATCH -p ---p---
#SBATCH --output=/home/klundert/cfdn/scripts/errout/%A_%a.out
#SBATCH --error=/home/klundert/cfdn/scripts/errout/%A_%a.err

#SBATCH --mail-type=END
#SBATCH --mail-user=---email---

module load 2022

cd /home/klundert/cfdn/scripts/
python fit_hcp_cf.py ---subject--- 16 ---data_portion--- $SLURM_ARRAY_TASK_ID

