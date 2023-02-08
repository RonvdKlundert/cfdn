#!/bin/bash
#SBATCH -t ---time---
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=---n---

#SBATCH -p ---p---
#SBATCH --output=/home/klundert/cfdn/scripts/errout2/%A_%a.out
#SBATCH --error=/home/klundert/cfdn/scripts/errout2/%A_%a.err

#SBATCH --mail-type=END
#SBATCH --mail-user=klundertclan@hotmail.com


module load 2022

cd /home/klundert/cfdn/scripts/
python fit_fsnative_hrf_gauss.py ---subject--- 16 ---data_portion---
