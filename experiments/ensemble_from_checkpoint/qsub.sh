#! /usr/bin/env bash

# Note: this doesn't use label smoothing

set -e
set -u

# --- SYSTEM ---
N_GPU=2  # args must be adjusted below if this is changed
UPDATE_FREQ=4
MAX_UPDATE=300000
GPU_TYPE=2080
NUM_PROC=20
MEM=12G
HOURS=48

if [ $# -lt 2 ]; then
   echo "Usage: ${0} JOB_NAME CKPT [FLAGS]"
   exit
fi

JOB_NAME=${1}
CKPT=${2}
shift
shift


DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt/${JOB_NAME}
if [ ! -d ${JOB_DIR} ]; then
    echo "${JOB_DIR} does not exist"
    exit
fi

CKPT_PATH="${JOB_DIR}/${CKPT}.pt"
if [ ! -f ${CKPT_PATH} ]; then
    echo "${CKPT_PATH} does not exist"
    exit
fi

ENSEMBLE_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_ensemble/${JOB_NAME}
mkdir -p ${ENSEMBLE_DIR}
JOB_SCRIPT=${ENSEMBLE_DIR}/job.sh
TRAIN="fairseq-train"

# Write training script
cat >${JOB_SCRIPT} <<EOL
#$ -cwd
#$ -V
#$ -w e
#$ -N ${JOB_NAME}_re
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
    --save-interval-updates 2000 \
    --lr-period-updates 2000 \
    --max-update 20000 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.00001 \
    --max-lr 0.001 \
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
    --max-update ${MAX_UPDATE} $@ \
    --log-format json \
    | tee ${ENSEMBLE_DIR}/train.log

EOL

chmod a+x ${JOB_SCRIPT}
QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
echo ${QSUB_CMD}
${QSUB_CMD}
#bash ${JOB_SCRIPT}

# eof
