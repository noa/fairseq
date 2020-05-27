#! /usr/bin/env bash

set -e
set -u

JOBS_DIR=/expscratch/nandrews/nmt/fairseq/jobs/mask_predict
# JOBS_DIR=../mask_predict

if [ $# -lt 1 ]; then
    echo "Usage: ${0} <CHECKPOINTS>"
    ls ${JOBS_DIR}
    exit
fi

CHECKPOINTS=`python ../scaling_nmt/join_ensemble_path.py ${JOBS_DIR} $@`
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

VALIDATE=`realpath ../../validate.py`

python ${VALIDATE} ${DATA_DIR} \
       --measure-calibration \
       --task translation_lev \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/valid \
       --path ${CHECKPOINTS} \
       --max-tokens 7000 \
       --num-workers 10

# eof
