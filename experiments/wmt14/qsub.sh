#! /usr/bin/env bash

set -e
set -u

# --- SYSTEM ---
N_GPU=4  # args must be adjusted below if this is changed
GPU_TYPE=2080
NUM_PROC=40
HOURS=48

# --- HPARAMS ---
ARCH=transformer_wmt_en_de

# --- OPTIMIZATION ---
UPDATE_FREQ=2
VAL_FREQ=20
PATIENCE=5

if [ $# -lt 3 ]; then
   echo "Usage: ${0} TASK JOB_NAME [FLAGS]"
   exit
fi

TASK=${1}
JOB_NAME=${2}
shift
shift

DATA_DIR=/expscratch/nandrews/fairseq/wmt14_${TASK}
if [ -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi
DATA=${DATA_DIR}/bin/wmt17_${TASK}
if [ -d "${DATA}" ]; then
    echo "${DATA} does not exist"
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
#$ -l h_rt=${HOURS}:00:00,num_proc=${NUM_PROC}
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
    ${DATA_DIR}/bin/wmt17_en_de \
    --save-dir ${JOB_DIR} \
    --seed ${SEED} \
    --arch ${ARCH} \
    --share-all-embeddings \
    --no-progress-bar \
    --keep-last-epochs 10 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 1e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 5000 \
    --warmup-init-lr 1e-7 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --validate-interval	${VAL_FREQ} \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --update-freq ${UPDATE_FREQ} \
    --fp16 \
    --patience ${PATIENCE}

EOL

chmod a+x ${JOB_SCRIPT}
qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU} ${JOB_SCRIPT}

# eof
