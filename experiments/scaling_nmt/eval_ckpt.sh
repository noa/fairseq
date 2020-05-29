#! /usr/bin/env bash

set -e
set -u

DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k

if [ $# -lt 3 ]; then
   echo "Usage: ${0} JOB_DIR JOB_NAME CKPT"
   exit
fi

EXPT_DIR=${1}
JOB_DIR=${2}
CKPT=${3}

FULL_EXPT_DIR=/expscratch/${USER}/nmt/fairseq/jobs/${EXPT_DIR}

if [ ! -d ${FULL_EXPT_DIR} ]; then
    echo "${FULL_EXPT_DIR} not found"
    ls /expscratch/${USER}/nmt/fairseq/jobs
    exit
fi

FULL_JOB_DIR=${FULL_EXPT_DIR}/${JOB_DIR}

if [ ! -d ${FULL_JOB_DIR} ]; then
    echo "${FULL_JOB_DIR} not found"
    ls -l ${FULL_EXPT_DIR}
    exit
fi

JOB_PATH=${FULL_JOB_DIR}/${CKPT}

if [ ! -f ${JOB_PATH} ]; then
    echo "${JOB_PATH} not found"
    ls -l ${FULL_JOB_DIR}
    exit
fi

# Note: without `--quiet`, this will print the translations and corresponding
# sequence- and token-level scores.
#GEN=/tmp/gen.out
GEN=$(mktemp /tmp/out.XXXXXXXXX)
fairseq-generate \
    ${DATA_DIR} \
    --path ${JOB_PATH} \
    --beam 4 --lenpen 0.6 --remove-bpe > ${GEN}

tail -n 3 ${GEN}

# See: https://github.com/pytorch/fairseq/issues/346

SYS="${GEN}.sys"
REF="${GEN}.ref"

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF

fairseq-score --sys ${SYS} --ref ${REF}

# eof
