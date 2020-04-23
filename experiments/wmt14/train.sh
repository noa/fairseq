#! /usr/bin/env bash

MAX_TOKENS=8192
UPDATE_FREQ=2
SEED=42
ARCH=transformer_wmt_en_de
DATA_DIR=/expscratch/nandrews/fairseq/wmt14_en2de
CKPT_DIR=/expscratch/${USER}/fairseq/wmt14_en2de/${1}

fairseq-train \
    ${DATA_DIR}/bin/wmt17_en_de \
    --save-dir ${CKPT_DIR} \
    --seed ${SEED} \
    --arch ${ARCH} \
    --share-decoder-input-output-embed \
    --keep-last-epochs 10 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
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
    --fp16 \
    --patience 4 \
    --best-checkpoint-metric val_loss

# eof
