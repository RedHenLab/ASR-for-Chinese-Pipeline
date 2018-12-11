#!/bin/bash
#
# Emulate cron by submitting jobs periodically

# Resources requested
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00

# Requeue in case of a node faiure
#SBATCH --requeue

# Job name and log
#SBATCH --job-name="Cron.sbatch -- submit jobs periodically"
#SBATCH --output=cron.log
#SBATCH --open-mode=append

# Submit the task
sbatch --quiet Chinese_ASR.slurm 5
sbatch --quiet Chinese_ASR.slurm 4
sbatch --quiet Chinese_ASR.slurm 3
sbatch --quiet Chinese_ASR.slurm 2
sbatch --quiet Chinese_ASR.slurm 1

# Recursive resubmit schedule -- for every hour, use "now+3600"
sbatch --quiet --begin=12:00 cron.sbatch

echo -e "\n\tcron.sbatch completed at `date`\n"
