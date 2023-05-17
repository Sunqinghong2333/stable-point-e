#!/usr/bin/env bash
set -x

PARTITION=${1}
srun -p ${PARTITION} --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=5 python -m point_e.vis