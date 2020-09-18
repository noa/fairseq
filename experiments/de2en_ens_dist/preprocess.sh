#! /usr/bin/env bash

SRC=/exp/nandrews/wmt16_en_de_bpe32k
TGT=/expscratch/nandrews/nmt/fairseq/data/wmt16_de_en_bpe32k

fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref $SRC/train.tok.clean.bpe.32000 \
    --validpref $SRC/newstest2013.tok.bpe.32000 \
    --testpref $SRC/newstest2014.tok.bpe.32000 \
    --destdir $TGT \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20

# eof
