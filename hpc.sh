#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --ntasks-per-node=10    # 10 mpi process each node
#SBATCH --mem=12000            # 12GB - in megabytes
#SBATCH --job-name="hpc job"
#SBATCH --output=log.txt
#SBATCH --mail-user=erik.s.sommer@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load fosscuda/2019a
module load TensorFlow/1.13.1-Python-3.7.2
cd src/
mpirun python3 train_hpc.py