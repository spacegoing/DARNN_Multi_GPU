#!/bin/bash
#PBS -q alloc-dt
#PBS -P chli
#PBS -l select=1:ncpus=36:ngpus=4:mpiprocs=32:mem=185GB
#PBS -l walltime=480:00:00

source spt
cd /home/chli4934/UsydCodeLab/phd/MrfEvent/code/src/

# python main.py --param_version=1 &
# python main.py --param_version=2 &
# python main.py --param_version=3 &
# python main.py --param_version=4 &
python main.py --param_version=5 &
python main.py --param_version=6 &
python main.py --param_version=7 &
