#! /usr/bin/env bash

set -e
set -u

JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt

if [ $# -lt 1 ]; then
    echo "Usage: ${0} <CHECKPOINTS>"
    ls ${JOBS_DIR}
    exit
fi

CHECKPOINTS=`python join_ensemble_path.py ${JOBS_DIR} $@`
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

VALIDATE=`realpath ../../validate.py`

python ${VALIDATE} ${DATA_DIR} \
       --measure-calibration \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/valid \
       --path ${CHECKPOINTS} \
       --max-tokens 7000 \
       --num-workers 10

# eof
