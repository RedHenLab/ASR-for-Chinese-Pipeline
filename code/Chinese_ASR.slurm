#!/bin/bash

# Resources
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH -p gpu -C gpuk40 --mem=100gb --gres=gpu:2

# Requeue in case of a node faiure
#SBATCH --requeue

# Title
#SBATCH --job-name="Chinese_ASR"

# Environment
module load singularity/2.5.1
module load cuda/7.5
export SINGULARITY_BINDPATH="/mnt"

# Working directory
cd /mnt/rds/redhen/gallina/Singularity/Chinese_Pipeline/code

# Task
singularity exec -e --nv ../Chinese_Pipeline.simg bash infer.sh $1

# Receipt
echo -e "\n\tThe Chinese ASR job completed at `date`\n"
