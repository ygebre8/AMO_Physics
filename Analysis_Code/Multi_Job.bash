# !/bin/bash
#PBS -S /bin/bash
#PBS -N YOSHI
#PBS -l nodes=1:ppn=1
#PBS -j oe
#PBS -m e

FILE="/home/becker/yoge8051/Research/Intensity_Sweep/*"
for entry in $FILE
do
    cd $entry
    qsub -q  photon13 /home/becker/yoge8051/Research/Queue_h.bash 
done
