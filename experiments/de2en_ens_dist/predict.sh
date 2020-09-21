#! /usr/bin/env bash

set -e
set -u

# Where we fetch pre-trained models
JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt

# Where we save the confidences
OUTPUT_DIR=/expscratch/${USER}/nmt/fairseq/jobs/teachers
mkdir -p ${OUTPUT_DIR}

if [ $# -lt 2 ]; then
    echo "Usage: ${0} <TOPK> <CKPT> <PATHS>"
    ls ${JOBS_DIR}
    exit
fi

echo "Additional args: $@"

TOP_K=${1}
CKPT=${2}
shift
shift

CHECKPOINTS=`python join_ensemble_path.py --checkpoint ${CKPT} ${JOBS_DIR} $@`
echo "Checkpoints: ${CHECKPOINTS}"
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_de_en_bpe32k

VALIDATE=`realpath ../../validate.py`
OUTPUT_FILE="$@_${TOP_K}_${CKPT}.h5"
OUTPUT_FILE=${OUTPUT_DIR}/"${OUTPUT_FILE// /_}"

echo "Output file: ${OUTPUT_FILE}"
echo "Top K: ${TOP_K}"

python ${VALIDATE} ${DATA_DIR} \
       --task translation \
       --criterion cross_entropy \
       --valid-subset ${DATA_DIR}/train \
       --full-dist-path ${OUTPUT_FILE} \
       --path ${CHECKPOINTS} \
       --max-tokens 8000 \
       --print-full-dist \
       --dist-top-k ${TOP_K} \
       --fp16 \
       --num-workers 10

# eof
