#!/bin/bash
#SBATCH --account=project_462000915
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1

ml purge

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3

#srun rocprof --hip-trace ./qubo
srun ./tests
