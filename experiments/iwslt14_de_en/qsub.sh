#! /usr/bin/env bash

set -e
set -u

# --- SYSTEM ---
N_GPU=2  # args must be adjusted below if this is changed
UPDATE_FREQ=4
GPU_TYPE=2080
NUM_PROC=20  # change this if GPU modified
MEM=12G
HOURS=48

# --- HPARAMS ---
ARCH=transformer_iwslt_de_en

# --- OPTIMIZATION ---
VAL_FREQ=5
PATIENCE=5
TOKENS=4096
MAX_STEPS=250000

if [ $# -lt 2 ]; then
   echo "Usage: ${0} JOB_NAME SEED [FLAGS]"
   exit
fi

JOB_NAME=${1}
SEED=${2}
shift
shift

DATA_DIR=/expscratch/nandrews/fairseq/iwslt14.tokenized.de-en
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

JOB_DIR=/expscratch/${USER}/fairseq/iwslt14_de_en/jobs/${JOB_NAME}
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
    ${DATA_DIR}/bin \
    --save-dir ${JOB_DIR} \
    --seed ${SEED} \
    --arch ${ARCH} \
    --share-decoder-input-output-embed \
    --no-progress-bar \
    --keep-last-epochs 10 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-7 \
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
