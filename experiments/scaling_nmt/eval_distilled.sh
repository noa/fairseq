#! /usr/bin/env bash

set -e
set -u

DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

if [ $# -lt 2 ]; then
    echo "Usage: ${0} JOB_NAME CKPT"
    ls /expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_distill
   exit
fi

JOB_NAME=${1}
CKPT=${2}
JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_distill/${JOB_NAME}

re='^[0-9]+$'

if ! [[ $CKPT =~ $re ]] ; then
    CKPT_PATH="${JOB_DIR}/checkpoint_${CKPT}.pt"
else
    AVG=`realpath ../../scripts/average_checkpoints.py`
    CKPT_PATH="${JOB_DIR}/checkpoint_avg.pt"
    if [ ! -f ${CKPT_PATH} ]; then
	python ${AVG} \
	       --inputs ${JOB_DIR} \
	       --num-epoch-checkpoints ${CKPT} \
	       --output ${CKPT_PATH}
    fi
fi

echo "Evaluating checkpoint at: ${CKPT_PATH}"

# Note: without `--quiet`, this will print the translations and corresponding
# sequence- and token-level scores.
GEN=/tmp/gen.out
fairseq-generate \
    ${DATA_DIR} \
    --source-lang en \
    --target-lang de \
    --path ${CKPT_PATH} \
    --beam 4 --lenpen 0.6 --remove-bpe > ${GEN}

tail -n 3 ${GEN}

# See: https://github.com/pytorch/fairseq/issues/346

SYS="${GEN}.sys"
REF="${GEN}.ref"

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF

fairseq-score --sys ${SYS} --ref ${REF}

# eof
