#! /usr/bin/env bash

set -e
set -u

# NOTE: Relative to the base configuration, we remove the regularization
# parameters, which were:
#
#  --dropout 0.3 --weight-decay 0.01
#
# Rationale: the teacher loss serves as a regularizer.

# NOTE: This version trains on raw, non-distilled training data. However, the
# original work always uses distilled translations.
#
# Rationale: The token-level distillation loss will exhibit the same kind of
# consistency as the 1-best translations would, while providing additional
# uncertainty information.
#
# TODO: Benchmark distilled translations vs. plain data.

# --- SYSTEM ---
N_GPU=2  # args must be adjusted below if this is changed
UPDATE_FREQ=6
MAX_TOKENS=3000
MAX_UPDATE=300000
GPU_TYPE=2080
NUM_PROC=20
MEM=12G
HOURS=48

TEACHER_DIR=/expscratch/nandrews/nmt/fairseq/jobs/teachers

if [ $# -lt 3 ]; then
   echo "Usage: ${0} JOB_NAME TEACHER TOPK TEMP WEIGHT [FLAGS]"
   exit
fi

JOB_NAME=${1}
TEACHER=${2}
TOPK=${3}
T=${4}
WEIGHT=${5}  # teacher weight
shift
shift
shift
shift
shift

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

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/mask_predict_distill/${JOB_NAME}_${T}_${WEIGHT}_${TEACHER}
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
    --save-dir ${JOB_DIR} \
    --seed ${SEED} \
    --ddp-backend=no_c10d \
    --teacher-pred ${TEACHER_FILE} \
    --teacher-top-k ${TOPK} \
    --distill-temperature ${T} \
    --teacher-weight ${WEIGHT} \
    --task translation_lev_with_teacher \
    --criterion nat_loss_with_teacher \
    --arch cmlm_transformer \
    --noise random_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --no-progress-bar \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens ${MAX_TOKENS} \
    --fp16 \
    --save-interval-updates 10000 \
    --keep-last-epochs 10 \
    --update-freq ${UPDATE_FREQ} \
    --max-update ${MAX_UPDATE} $@

EOL

chmod a+x ${JOB_SCRIPT}
QSUB_CMD="qsub -q gpu.q@@${GPU_TYPE} -l gpu=${N_GPU},mem_free=${MEM},h_rt=${HOURS}:00:00,num_proc=${NUM_PROC} ${JOB_SCRIPT}"
echo ${QSUB_CMD}
${QSUB_CMD}

# eof
