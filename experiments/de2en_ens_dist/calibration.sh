#! /usr/bin/env bash

set -e
set -u

JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/de2en_ens_distill

if [ $# -lt 2 ]; then
    echo "Usage: ${0} <SPLIT> <JOBS>"
    ls ${JOBS_DIR}
    exit
fi

SPLIT=$1
shift

CHECKPOINTS=`python join_ensemble_path.py ${JOBS_DIR} $@`
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_de_en_bpe32k

echo "Checkpoints: ${CHECKPOINTS}"

VALIDATE=`realpath ../../validate.py`

python ${VALIDATE} ${DATA_DIR} \
       --measure-calibration \
       --source-lang de \
       --target-lang en \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/${SPLIT} \
       --path ${CHECKPOINTS} \
       --max-tokens 4096 \
       --num-workers 10

# eof
