#!/bin/bash
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=8

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${CPU_BIND} ./select_gpu bin/qubo data/input.json
rm -rf ./select_gpu
