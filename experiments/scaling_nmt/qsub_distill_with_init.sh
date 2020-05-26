#! /usr/bin/env bash

set -e
set -u

# NOTE: In addition to epoch checkpoints, we save every 1000 steps.
#       We keep the last 10 such checkpoints.
#
# NOTE: Dropout is disabled
#
# NOTE: We initialize from an average checkpoint rather than the last.
#
# NOTE: float16 is disabled

# --- SYSTEM ---
N_GPU=4  # args must be adjusted below if this is changed
NUM_PROC=40
GPU_TYPE=2080
MEM=12G
HOURS=48

# --- BATCHING ---
UPDATE_FREQ=4
MAX_TOKENS=3000
WARMUP_UPDATE=2000

TEACHER_DIR=/expscratch/nandrews/nmt/fairseq/jobs/teachers

if [ $# -lt 6 ]; then
   echo "Usage: ${0} JOB_NAME TEACHER TOPK TEMP WEIGHT INIT MAX_UPDATE DIVERGENCE [FLAGS]"
   exit
fi

JOB_NAME=${1}
TEACHER=${2}
TOPK=${3}
T=${4}
WEIGHT=${5}  # teacher weight
INIT_JOB=${6}
MAX_UPDATE=${7}
DIVERGENCE=${8}
shift
shift
shift
shift
shift
shift
shift
shift

echo "Extra arguments: $@"
echo "Temperature: ${T}"
echo "Distillation loss weight: ${WEIGHT}"
echo "Divergence: ${DIVERGENCE}"

DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt16_en_de_bpe32k
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

TEACHER_FILE="${TEACHER_DIR}/${TEACHER}"
if [ ! -f "${TEACHER_FILE}" ]; then
    echo "${TEACHER} not found"
    ls -l ${TEACHER_DIR}
    exit
fi

INIT_DIR=/expscratch/nandrews/nmt/fairseq/jobs/scaling_nmt
INIT_FILE="${INIT_DIR}/${INIT_JOB}/checkpoint_avg.pt"
if [ ! -f "${INIT_FILE}" ]; then
    echo "${INIT_FILE} not found"
    ls -l ${INIT_DIR}
    exit
fi
echo "Init file: ${INIT_FILE}"

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_distill/${JOB_NAME}_${T}_${WEIGHT}_${TEACHER}_${MAX_UPDATE}
mkdir -p ${JOB_DIR}
JOB_SCRIPT=${JOB_DIR}/job.sh

echo "${JOB_DIR}"
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
    ${DATA_DIR} \
    --task translation_with_teacher \
    --reset-optimizer \
    --reset-lr-scheduler \
    --reset-dataloader \
    --restore-file ${INIT_FILE} \
    --teacher-pred ${TEACHER_FILE} \
    --teacher-top-k ${TOPK} \
    --distill-loss-type combined \
    --distill-divergence ${DIVERGENCE} \
    --distill-temperature ${T} \
    --teacher-weight ${WEIGHT} \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates ${WARMUP_UPDATE} \
    --warmup-init-lr 1e-07 \
    --weight-decay 0.0 \
    --dropout 0.0 \
    --criterion distillation_cross_entropy \
    --no-progress-bar \
    --save-dir ${JOB_DIR} \
    --max-tokens ${MAX_TOKENS} \
    --keep-last-epochs 10 \
    --update-freq ${UPDATE_FREQ} \
    --keep-interval-updates 1000 \
    --keep-interval-updates 10 \
    --max-update ${MAX_UPDATE} $@

EOL

chmod a+x ${JOB_SCRIPT}
QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
echo ${QSUB_CMD}
${QSUB_CMD}

# eof
