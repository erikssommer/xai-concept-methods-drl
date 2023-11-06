#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=200G
#SBATCH --output=log.txt
#SBATCH --mail-user=erik.s.sommer@ntnu.no
#SBATCH --mail-type=ALL

echo "Running script"

module purge
module load Anaconda3/2022.10
conda activate go-tf
cd src/
python train.py
