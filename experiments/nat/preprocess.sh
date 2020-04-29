#! /usr/bin/env bash

SRC=/exp/nandrews/nmt_datasets/distill/wmt14_ende_distill
TGT=/expscratch/nandrews/nmt/fairseq/data/wmt14_ende_distill

fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $SRC/train.en-de \
    --validpref $SRC/valid-repeat.en-de \
    --testpref $SRC/test.en-de \
    --destdir $TGT \
    --joined-dictionary \
    --workers 20
    
#--srcdict $SRC/wmt14.bpe.codes

# eof
