#! /usr/bin/env bash

set -e
set -u

# NOTE: Dropout is disabled
# NOTE: We initialize from an average checkpoint rather than the last.
# NOTE: float16 is disabled

# --- SYSTEM ---
UPDATE_FREQ=6
MAX_TOKENS=3000
MAX_UPDATE=20000

TEACHER_DIR=/expscratch/nandrews/nmt/fairseq/jobs/teachers

if [ $# -lt 6 ]; then
   echo "Usage: ${0} JOB_NAME TEACHER TOPK TEMP WEIGHT [FLAGS]"
   exit
fi

JOB_NAME=${1}
TEACHER=${2}
TOPK=${3}
T=${4}
WEIGHT=${5}  # teacher weight
INIT_JOB=${6}
shift
shift
shift
shift
shift
shift

echo "Extra arguments: $@"
echo "Temperature: ${T}"
echo "Distillation loss weight: ${WEIGHT}"

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

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/scaling_nmt_distill/${JOB_NAME}_${T}_${WEIGHT}_${TEACHER}
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
    --distill-temperature ${T} \
    --teacher-weight ${WEIGHT} \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt \
    --warmup-updates 1000 \
    --warmup-init-lr 1e-07 \
    --weight-decay 0.0 \
    --dropout 0.0 \
    --criterion distillation_cross_entropy \
    --no-progress-bar \
    --save-dir ${JOB_DIR} \
    --max-tokens ${MAX_TOKENS} \
    --keep-last-epochs 10 \
    --update-freq ${UPDATE_FREQ} \
    --max-update ${MAX_UPDATE} $@

EOL

chmod a+x ${JOB_SCRIPT}
#QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
#echo ${QSUB_CMD}
#${QSUB_CMD}
bash ${JOB_SCRIPT}

# eof
