#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1    # Request any available GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # There are 32 CPU cores on v100 Cedar GPU nodes
#SBATCH --mem=64G              # Request the full memory of the node
#SBATCH --time=1:55:00

# ham10000
#python train.py --config ./configs/ham10000_train.yaml
python centralized_HAM.py

# kvasir-instrument
#python train.py --config ./configs/kvasir-instrument_train.yaml
