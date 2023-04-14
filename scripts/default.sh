#!/bin/bash
# Submission script for 
#SBATCH --job-name=dialign
#SBATCH --time=24:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=8192 # 8GB
#SBATCH --partition=gpu
#
#SBATCH --mail-user=vinh.tong@ipvs.uni-stuttgart.de
#SBATCH --mail-type=ALL
#
#SBATCH --output=out.txt

#pip install -e . 

python -u src/align.py > logt.txt 

