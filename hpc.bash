#!/bin/bash

module purge
module load Anaconda3/2022.10
conda activate go-tf

sbatch job.slurm