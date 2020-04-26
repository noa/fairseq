#! /usr/bin/env bash

set -u

TEXT=/expscratch/nandrews/fairseq/iwslt14.tokenized.de-en
DEST=/expscratch/nandrews/fairseq/iwslt14.tokenized.de-en
mkdir -p ${DEST}
fairseq-preprocess --source-lang de \
		   --target-lang en \
		   --trainpref ${TEXT}/train \
		   --validpref ${TEXT}/valid \
		   --testpref ${TEXT}/test \
		   --destdir ${DEST}/bin \
		   --thresholdtgt 0 \
		   --thresholdsrc 0 \
		   --workers 20

# eof
