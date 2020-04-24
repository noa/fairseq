#! /usr/bin/env bash

TEXT=/exp/nandrews/nmt_datasets/wmt16_en_de
DEST=/expscratch/nandrews/fairseq/wmt16_en_de
mkdir -p ${DEST}
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir ${DEST}/bin/wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary

# eof
