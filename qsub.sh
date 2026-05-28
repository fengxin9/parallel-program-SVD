#!/bin/sh
#PBS -N qsub
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=1

/usr/local/bin/pssh -h $PBS_NODEFILE mkdir -p /home/${USER} 1>&2
scp master_ubss1:/home/${USER}/svd/main /home/${USER} 1>&2
scp -r master_ubss1:/home/${USER}/svd/files/ /home/${USER}/ 1>&2
/usr/local/bin/pscp -h $PBS_NODEFILE /home/${USER}/main /home/${USER} 1>&2

if [ -n "$SVD_SEED" ]; then
    /usr/local/bin/mpiexec -np 1 -machinefile $PBS_NODEFILE /home/${USER}/main "$SVD_SEED"
else
    /usr/local/bin/mpiexec -np 1 -machinefile $PBS_NODEFILE /home/${USER}/main
fi

rm /home/${USER}/main
scp -r /home/${USER}/files/ master_ubss1:/home/${USER}/svd/ 2>&1
rm -r /home/${USER}/files/
