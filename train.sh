#!/usr/bin/env bash
set -x

GPUS=${1}
JOB_NAME=${2}
PARTITION=${3}
COND=${4}
CPUS_PER_TASK=${5}
WORK_ROOT=${6}
CONFIG=${7}


START_TIME=`date +%m%d-%H%M%S`
WORK_DIR=$WORK_ROOT/$JOB_NAME-$START_TIME
LOG_FILE=$WORK_DIR/logs
mkdir -p $LOG_FILE
if [ ${GPUS} -gt 8 ];then
    GPUS_PER_NODE=8
else
    GPUS_PER_NODE=${GPUS}
fi



srun -p ${PARTITION} --job-name=${JOB_NAME} --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} --ntasks-per-node=${GPUS_PER_NODE}  --cpus-per-task=${CPUS_PER_TASK} \
    python -m point_e.train --config=${CONFIG} --save-root=${WORK_DIR} --num-workers=${CPUS_PER_TASK} --log-dir=${LOG_FILE} --cond ${COND} \
    2>&1 | tee $WORK_DIR/log.txt > /dev/null &
