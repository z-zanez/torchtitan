#!/bin/bash
#SBATCH --job-name=torchtitan_single_node
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=train

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_1b.toml"}

srun python -m torchtitan.train --job.config_file ${CONFIG_FILE}