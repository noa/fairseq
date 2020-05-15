#! /usr/bin/env bash

# Note: this doesn't use label smoothing

set -e
set -u

# --- SYSTEM ---
N_GPU=2  # args must be adjusted below if this is changed
UPDATE_FREQ=4
GPU_TYPE=2080
NUM_PROC=20
MEM=12G
HOURS=48

if [ $# -lt 7 ]; then
    echo "Usage: ${0} JOB_NAME CKPT SAVE_STEPS PERIOD STEPS MIN_LR MAX_LR"
    echo "Got $# arguments"
    exit
fi

JOB_NAME=${1}  # e.g. e1b
CKPT=${2}  # e.g. checkpoint60
SAVE_STEPS=${3}  # e.g. 1999
PERIOD=${4}  # e.g. 2000
STEPS=${5}  # e.g. 20000
MIN_LR=${6}  # e.g. 0.00001
MAX_LR=${7}  # e.g. 0.001


DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt/${JOB_NAME}
if [ ! -d ${JOB_DIR} ]; then
    echo "${JOB_DIR} does not exist"
    ls /expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt/
    exit
fi

CKPT_PATH="${JOB_DIR}/${CKPT}.pt"
if [ ! -f ${CKPT_PATH} ]; then
    echo "${CKPT_PATH} does not exist"
    ls ${JOB_DIR}
    exit
fi

ENSEMBLE_JOB_NAME=${JOB_NAME}_${CKPT}_${SAVE_STEPS}_${PERIOD}_${STEPS}_${MIN_LR}_${MAX_LR}
ENSEMBLE_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_ensemble/${ENSEMBLE_JOB_NAME}
mkdir -p ${ENSEMBLE_DIR}
JOB_SCRIPT=${ENSEMBLE_DIR}/job.sh
TRAIN="fairseq-train"

# Write training script
cat >${JOB_SCRIPT} <<EOL
#$ -cwd
#$ -V
#$ -w e
#$ -N ${ENSEMBLE_JOB_NAME}
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
    ${DATA_DIR} \
    --restore-file ${CKPT_PATH} \
    --save-dir ${ENSEMBLE_DIR} \
    --arch transformer_wmt_en_de \
    --share-all-embeddings \
    --reset-lr-scheduler \
    --reset-optimizer \
    --reset-meters \
    --log-interval 50 \
    --save-interval-updates ${SAVE_STEPS} \
    --lr-period-updates ${PERIOD} \
    --max-update ${STEPS} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr ${MIN_LR} \
    --max-lr ${MAX_LR} \
    --lr-scheduler cosine \
    --t-mult 1 \
    --lr-shrink 1 \
    --warmup-updates 0 \
    --weight-decay 0.0 \
    --dropout 0.1 \
    --criterion cross_entropy \
    --no-progress-bar \
    --max-tokens 4096 \
    --fp16 \
    --keep-last-epochs 10 \
    --update-freq ${UPDATE_FREQ} \
    --log-format json \
    | tee ${ENSEMBLE_DIR}/train.log

EOL

chmod a+x ${JOB_SCRIPT}
QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
echo ${QSUB_CMD}
${QSUB_CMD}
#bash ${JOB_SCRIPT}

# eof
