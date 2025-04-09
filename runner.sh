#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00

python HAM_V1_FP.py   #Change the file name accordingly
