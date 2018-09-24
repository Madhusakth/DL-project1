# DL-project1
Log into TACC maverick: 
ssh username@maverick.tacc.utexas.edu

To load python3, cuda, cudnn and tensorflow:
module reset

module load gcc/4.9.1 cuda/8.0 cudnn/5.1 python3/3.5.2 tensorflow-gpu/1.0.0


To submit a job: 
sbatch cmd.sh
To check the queue: 
showq -u

To remove a particular job:
scancel (job id)
