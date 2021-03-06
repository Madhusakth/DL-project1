#!/bin/sh
#SBATCH -J deeplearnproj1           # Job name
#SBATCH -o experiment.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p vis                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 4:00:00              # Run time (hh:mm:ss) - 3.5 hours



python3 run_test.py
