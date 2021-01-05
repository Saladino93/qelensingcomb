#!/bin/bash
#SBATCH --nodes=30
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --output=/scratch/r/rbond/omard/CORI17112020/mpioutput/mpi_output_%j.txt
#SBATCH --cpus-per-task=2


cd $SLURM_SUBMIT_DIR

export DISABLE_MPI=false

module load autotools
module load intelmpi
module load intelpython3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun python extract_biases.py config.yaml  
