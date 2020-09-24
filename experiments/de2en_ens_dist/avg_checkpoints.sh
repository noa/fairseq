#! /usr/bin/env bash

set -e
set -u

JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/de2en

if [ $# -lt 2 ]; then
    echo "Usage: ${0} JOB_NAME LAST_N"
    ls ${JOBS_DIR}
   exit
fi

JOB_NAME=${1}
JOB_DIR=${JOBS_DIR}/${JOB_NAME}
LAST_N=${2}

AVG=`realpath ../../scripts/average_checkpoints.py`

python ${AVG} \
    --inputs ${JOB_DIR} \
    --num-epoch-checkpoints ${LAST_N} \
    --output ${JOB_DIR}/checkpoint_avg.pt

# eof
