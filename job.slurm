#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=2     # 2 CPU cores
#SBATCH --mem=5G              # 5 gigabytes memory
#SBATCH --output=log.txt      # Log file
#SBATCH --mail-user=erik.s.sommer@ntnu.no
#SBATCH --mail-type=ALL

echo "Hello IDUN"

module purge
module load Anaconda3/2022.10
conda activate go-tf
cd src/
python train.py