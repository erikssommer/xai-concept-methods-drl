#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1              # 1 compute nodes
#SBATCH --ntasks-per-node=21   # 21 mpi process each node
#SBATCH --mem=50000            # 50GB - in megabytes
#SBATCH --gres=gpu:p100:1      # 1 tesla gpu
#SBATCH --job-name="hpc-job"
#SBATCH --output=log.txt
#SBATCH --mail-user=erik.s.sommer@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
cd src/
mpirun python train_hpc.py
