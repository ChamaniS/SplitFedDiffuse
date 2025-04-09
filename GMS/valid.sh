#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1    # Request any available GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # There are 32 CPU cores on v100 Cedar GPU nodes
#SBATCH --mem=64G              # Request the full memory of the node
#SBATCH --time=8:55:00

# ham10000
python valid.py --config ./configs/ham10000_valid.yaml

# kvasir-instrument
#python valid.py --config ./configs/kvasir-instrument_valid.yaml
