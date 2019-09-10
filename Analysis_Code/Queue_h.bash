# !/bin/bash
#PBS -S /bin/bash
#PBS -N YOSHI
#PBS -l nodes=1:ppn=12
#PBS -j oe
#PBS -m e

#$JOB_DIR=$1
JOB_DIR="/home/becker/yoge8051/Research/Circular/3-photon-process/"
module purge
module load intel openmpi hdf5 boost cmake blas gsl
cd $JOB_DIR
#cd "$PBS_O_WORKDIR" || exit $?
hostname
pwd

mpiexec -n 12 /home/becker/yoge8051/Research/TDSE/bin/TDSE >> run.log
