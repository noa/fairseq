#!/bin/sh

set -euxo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: ${0} N_TO_AVG JOB_NAME1 [JOB_NAME2] [...]"
    ls /expscratch/${USER}/nmt/fairseq/jobs/de2en
    exit
fi

N=$1
shift
JOBS=$@

echo "Averaging last ${N} checkpoints for each expert"
echo "Experts: ${JOBS}"

declare -A CKPTS

for JOB_NAME in $@; do
    JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/de2en/${JOB_NAME}
    AVG=`realpath ../../scripts/average_checkpoints.py`
    CKPT="${JOB_DIR}/checkpoint_avg.pt"
    CKPTS[${JOB_NAME}]="${CKPT}"
    if [ ! -f ${CKPT} ]; then
       python ${AVG} \
	      --inputs ${JOB_DIR} \
	      --num-epoch-checkpoints 10 \
	      --output ${CKPT}
    fi
done

JOINED_PATH=""
for CKPT in "${!CKPTS[@]}"; do
    echo "$CKPT - ${CKPTS[$CKPT]}"
    JOINED_PATH="${CKPTS[$CKPT]}:${JOINED_PATH}"
done

echo "${JOINED_PATH}"

DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_de_en_bpe32k

# Note: without `--quiet`, this will print the translations and corresponding
# sequence- and token-level scores.
GEN=/tmp/gen.out
CMD=`realpath ../../fairseq_cli/generate.py`
echo "${CMD}"
python ${CMD} ${DATA_DIR} --path ${JOINED_PATH::-1} --beam 4 --lenpen 0.6 --remove-bpe > ${GEN}

tail -n 3 ${GEN}

# See: https://github.com/pytorch/fairseq/issues/346

SYS="${GEN}.sys"
REF="${GEN}.ref"

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF

fairseq-score --sys ${SYS} --ref ${REF}

# eof
