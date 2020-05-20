#! /usr/bin/env bash

set -e
set -u

JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt

if [ $# -lt 1 ]; then
    echo "Usage: ${0} JOB_NAME"
    ls ${JOBS_DIR}
    exit
fi

JOB_NAME=${1}
JOB_DIR=${JOBS_DIR}/${JOB_NAME}
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

VALIDATE=`realpath ../../validate.py`

python ${VALIDATE} ${DATA_DIR} \
       --measure-calibration \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/valid \
       --path ${JOB_DIR}/checkpoint_last.pt \
       --max-tokens 7000 \
       --num-workers 10

# eof
