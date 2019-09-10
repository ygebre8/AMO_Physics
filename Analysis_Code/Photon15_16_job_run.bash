module purge
module load intel openmpi hdf5 boost cmake blas gsl

FILE="/home/becker/yoge8051/AMO_Physics/Intensity_Sweep/B/*"
for entry in $FILE
do
    cd $entry
    pwd
    mpiexec -n 24 /home/becker/yoge8051/Research/TDSE/bin/TDSE >> run.log
done
