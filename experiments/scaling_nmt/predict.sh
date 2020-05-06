#! /usr/bin/env bash

set -e
set -u

if [ $# -lt 1 ]; then
   echo "Usage: ${0} JOB_NAME"
   exit
fi

JOB_NAME=${1}
JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt/${JOB_NAME}
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

VALIDATE=`realpath ../../validate.py`
OUTPUT_FILE="${JOB_DIR}/train_${JOB_NAME}_dist.pkl"

echo "Output file: ${OUTPUT_FILE}"

python ${VALIDATE} ${DATA_DIR} \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/train \
       --full-dist-path ${OUTPUT_FILE} \
       --path ${JOB_DIR}/checkpoint_last.pt \
       --max-tokens 7000 \
       --print-full-dist \
       --fp16 \
       --num-workers 10

# eof
