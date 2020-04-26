#! /usr/bin/env bash

if [ $# -lt 1 ]; then
   echo "Usage: ${0} JOB_NAME"
   exit
fi

JOB_NAME=${1}
JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt/${JOB_NAME}

AVG=`realpath ../../scripts/average_checkpoints`

python ${AVG} \
    --inputs ${JOB_DIR} \
    --num-epoch-checkpoints 10 \
    --output /tmp/avg_checkpoint.pt

DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

fairseq-generate \
    ${DATA_DIR} \
    --path /tmp/avg_checkpoint.pt \
    --beam 4 --lenpen 0.6 --remove-bpe

# eof
