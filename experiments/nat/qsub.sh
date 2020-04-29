#! /usr/bin/env bash

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
   echo "Usage: ${0} JOB_NAME SEED [FLAGS]"
   exit
fi

TASK=en_de
JOB_NAME=${1}
SEED=${2}
shift
shift


DATA_DIR=/expscratch/nandrews/nmt/fairseq/data/wmt14_ende_distill
if [ ! -d "${DATA_DIR}" ]; then
    echo "${DATA_DIR} does not exist"
    exit
fi

JOB_DIR=/expscratch/${USER}/nmt/fairseq/jobs/nat/${JOB_NAME}
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
    ${DATA_DIR} \
    --save-dir ${JOB_DIR} \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
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
