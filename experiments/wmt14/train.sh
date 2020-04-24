#! /usr/bin/env bash

MAX_TOKENS=4096
TASK=en_de
UPDATE_FREQ=2
SEED=42
ARCH=transformer_wmt_en_de
DATA_DIR=/expscratch/nandrews/fairseq/wmt14_${TASK}
CKPT_DIR=/expscratch/${USER}/fairseq/wmt14_${TASK}/${1}

fairseq-train \
    ${DATA_DIR}/bin/wmt14_${TASK} \
    --save-dir ${CKPT_DIR} \
    --seed ${SEED} \
    --arch ${ARCH} \
    --share-decoder-input-output-embed \
    --keep-last-epochs 5 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 5000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens ${MAX_TOKENS} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --update-freq ${UPDATE_FREQ} \
    --patience 4 \
    --fp16

# eof
