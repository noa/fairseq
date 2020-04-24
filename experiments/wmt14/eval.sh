#! /usr/bin/env bash

set -u
set -e

if [ $# -lt 1 ]; then
   echo "Usage: ${0} JOB_NAME"
   exit
fi

JOB_NAME=${1}
TASK="en_de"
DATA_DIR=/expscratch/nandrews/fairseq/wmt14_${TASK}
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi
JOB_DIR=/expscratch/${USER}/fairseq/wmt14_${TASK}/jobs/${JOB_NAME}
ckpt=${JOB_DIR}/checkpoint_best.pt
subset="test"
  
fairseq-generate ${DATA_DIR}/bin/wmt14_en_de  \
		 --path $ckpt \
		 --gen-subset $subset \
		 --beam 4 \
		 --batch-size 128 \
		 --remove-bpe \
		 --lenpen 0.6

# eof
