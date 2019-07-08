#!/usr/bin/bash
#PBS -l modes=1:ppn=1
#PBS -N YOSHI
#PBS -j oe
#PBS -q photon6

FILE="/home/becker/yoge8051/Research/L_Absorber_Material/*"

for entry in $FILE
do 
    echo $entry >> run.log
    cd $entry
    python /home/becker/yoge8051/Research/Analysis_Code/HHG_Comp.py $entry >> run.log  
done 


