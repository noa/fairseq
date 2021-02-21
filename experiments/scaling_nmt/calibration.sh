#! /usr/bin/env bash

set -e
set -u

#JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt

if [ $# -lt 3 ]; then
    echo "Usage: ${0} <SPLIT> <JOBS_DIR> <JOBS>"
    ls ${JOBS_DIR}
    exit
fi

SPLIT=$1
JOBS_DIR=$2
shift
shift

CHECKPOINTS=`python join_ensemble_path.py ${JOBS_DIR} $@`
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

echo "Checkpoints: ${CHECKPOINTS}"

VALIDATE=`realpath ../../validate.py`

python ${VALIDATE} ${DATA_DIR} \
       --measure-calibration \
       --source-lang en \
       --target-lang de \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/${SPLIT} \
       --path ${CHECKPOINTS} \
       --max-tokens 4096 \
       --num-workers 10

# eof
