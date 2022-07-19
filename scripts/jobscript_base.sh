#!/bin/bash
#SBATCH -t 0:10:00
#SBATCH -n 16
#SBATCH -p short
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err


#SBATCH --mail-type=END
#SBATCH --mail-user=klundertclan@hotmail.com

cd /home/klundert/cfdn/scripts/
python 1_CF_fit_hcp.py ---sub--- ---n_jobs--- ---slice_n--- ---chunk_n---