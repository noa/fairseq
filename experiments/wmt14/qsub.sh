#!/bin/bash
##! /usr/bin/env bash

set -e
set -u

# --- SYSTEM ---
N_GPU=2  # args must be adjusted below if this is changed
UPDATE_FREQ=4
GPU_TYPE=2080
NUM_PROC=40
MEM=12G
HOURS=48

# --- HPARAMS ---
ARCH=transformer_wmt_en_de

# --- OPTIMIZATION ---
VAL_FREQ=5
PATIENCE=5
TOKENS=4096
MAX_STEPS=250000

if [ $# -lt 3 ]; then
   echo "Usage: ${0} TASK JOB_NAME SEED [FLAGS]"
   exit
fi

TASK=${1}
JOB_NAME=${2}
SEED=${3}
shift
shift
shift

DATA_DIR=/expscratch/nandrews/fairseq/wmt14_${TASK}
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

JOB_DIR=/expscratch/${USER}/fairseq/wmt14_${TASK}/jobs/${JOB_NAME}
mkdir -p ${JOB_DIR}
JOB_SCRIPT=${JOB_DIR}/job.sh

TRAIN="fairseq-train"

# Write training script
cat >${JOB_SCRIPT} <<EOL
#$ -cwd
#$ -V
#$ -w e
#$ -N ${JOB_NAME}
#$ -m bea
#$ -j y
#$ -o ${JOB_DIR}/out
#$ -e ${JOB_DIR}/err

# Stop on error
set -e
set -u
set -f

module load cuda10.1/toolkit
module load cuda10.1/blas
module load cudnn/7.6.3_cuda10.1
module load nccl/2.4.7_cuda10.1
export MKL_SERVICE_FORCE_INTEL=1

fairseq-train \
    ${DATA_DIR}/bin/wmt14_en_de \
    --save-dir ${JOB_DIR} \
    --seed ${SEED} \
    --arch ${ARCH} \
    --share-decoder-input-output-embed \
    --no-progress-bar \
    --keep-last-epochs 10 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.0007 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 5000 \
    --warmup-init-lr 1e-9 \
    --min-lr 1e-09 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens ${TOKENS} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --validate-interval	${VAL_FREQ} \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --update-freq ${UPDATE_FREQ} \
    --fp16 \
    --max-update ${MAX_STEPS}

EOL

chmod a+x ${JOB_SCRIPT}
QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
echo ${QSUB_CMD}
${QSUB_CMD}


# eof
