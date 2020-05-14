#! /usr/bin/env bash

set -e
set -u

JOBS_DIR=/expscratch/${USER}/nmt/fairseq/jobs/mask_predict

if [ $# -lt 1 ]; then
    echo "Usage: ${0} JOB_NAME"
    ls ${JOBS_DIR}
    exit
fi

JOB_NAME=${1}
JOB_DIR=${JOBS_DIR}/${JOB_NAME}

AVG=`realpath ../../scripts/average_checkpoints.py`

# The paper says:
#
# "We trained all models for 300k steps, measured the validation loss
# at the end of each epoch, and averaged the 5 best checkpoints"
#
# Note that below we simply average the *last* 5 checkpoints, which
# may or may not be the *best* checkpoints (but most likely are).

python ${AVG} \
    --inputs ${JOB_DIR} \
    --num-epoch-checkpoints 5 \
    --output /tmp/avg_checkpoint.pt

#DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k
DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt14_ende_distill

# Note: without `--quiet`, this will print the translations and corresponding
# sequence- and token-level scores.
GEN=/tmp/gen.out
fairseq-generate \
    ${DATA_DIR} \
    --path /tmp/avg_checkpoint.pt \
    --task translation_lev \
    --max-sentences 20 \
    --iter-decode-with-beam 5 \
    --iter-decode-max-iter 10 \
    --iter-decode-force-max-iter \
    --beam 5 --remove-bpe > ${GEN}

tail -n 3 ${GEN}

# See: https://github.com/pytorch/fairseq/issues/346

SYS="${GEN}.sys"
REF="${GEN}.ref"

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF

fairseq-score --sys ${SYS} --ref ${REF}

# eof
