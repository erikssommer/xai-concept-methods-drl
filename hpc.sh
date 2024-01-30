#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=4-00:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --output=log.txt
#SBATCH --mail-user=erik.s.sommer@ntnu.no
#SBATCH --mail-type=ALL

echo "Running script for training on HPC"

module purge
module load Anaconda3/2022.10
conda activate go-tf
cd src/
mpirun python train_hpc.py